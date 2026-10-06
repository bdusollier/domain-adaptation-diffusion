#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import functools
import gc
import json
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from datasets import load_dataset, load_from_disk
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, T5EncoderModel, CLIPTextModel

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxControlNetPipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from src.flux.mobilenet_controlnet import MobileNetFluxControlNetModel

if is_wandb_available():
    import wandb

logger = get_logger(__name__)


def _grad_norm(module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.data.float().pow(2).sum().item()
    return math.sqrt(total)


def _weight_norm(module) -> float:
    total = 0.0
    if isinstance(module, torch.Tensor):
        params = [module]
    else:
        params = module.parameters()
    for p in params:
        total += p.data.float().pow(2).sum().item()
    return math.sqrt(total)


def log_validation(vae, transformer, controlnet, args, accelerator, weight_dtype, step, is_final_validation=False):
    logger.info("Running validation... ")
    from diffusers import FluxControlNetPipeline
    from PIL import Image as PILImage

    controlnet_model = accelerator.unwrap_model(controlnet)

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, use_fast=False,
    )
    tokenizer_2 = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant,
        torch_dtype=weight_dtype,
    )
    text_encoder_2 = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant,
        torch_dtype=weight_dtype,
    )

    pipeline = FluxControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        transformer=transformer,
        controlnet=controlnet_model,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        torch_dtype=weight_dtype,
    )

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    image_logs = []
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type, dtype=weight_dtype)

    for validation_prompt, validation_image in zip(validation_prompts, validation_images):
        validation_image = PILImage.open(validation_image).convert("RGB")

        interpolation = transforms.InterpolationMode.LANCZOS
        transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=interpolation),
            transforms.CenterCrop(args.resolution),
        ])
        validation_image = transform(validation_image)

        images = []
        for _ in range(args.num_validation_images):
            with autocast_ctx:
                image = pipeline(
                    prompt=validation_prompt,
                    control_image=validation_image,
                    num_inference_steps=20,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                ).images[0]
            images.append(image)

        image_logs.append(
            {"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt}
        )

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]
                formatted_images = [np.asarray(validation_image)]
                for image in images:
                    formatted_images.append(np.asarray(image))
                formatted_images = np.stack(formatted_images)
                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]
                formatted_images.append(wandb.Image(validation_image, caption="Controlnet conditioning"))
                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)
            tracker.log({tracker_key: formatted_images})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    del pipeline, text_encoder, text_encoder_2, tokenizer, tokenizer_2
    gc.collect()
    torch.cuda.empty_cache()

    return image_logs


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            make_image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    model_description = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with MobileNetV3 T2I-Adapter conditioning for FLUX.1.
{img_str}
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="openrail++",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )
    tags = [
        "flux",
        "text-to-image",
        "diffusers",
        "controlnet",
        "mobilenet-controlnet",
        "diffusers-training",
    ]
    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="FLUX ControlNet training with MobileNetV3 T2I-Adapter.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained FLUX model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to an existing controlnet model to initialize weights from.",
    )
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=3.0,
        help="Flow matching time-shift parameter for FLUX (default 3.0 for 1024x1024). Set to 1.0 for uniform schedule.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-flux-mobilenet",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="The resolution for input images, all the images in the train/validation dataset will be resized to this resolution.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint of the training state every X updates.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Max number of checkpoints to store.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Whether training should be resumed from a previous checkpoint.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help='The scheduler type to use.',
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=100, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of hard resets of the lr in cosine_with_restarts scheduler.")
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="The log directory for TensorBoard/WandB.",
    )
    parser.add_argument("--allow_tf32", action="store_true", help="Whether or not to allow TF32 on Ampere GPUs.")
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help='The integration to report the results and logs to.',
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision.",
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help="Save more memory by using setting grads to None instead of zero.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the Dataset (from the HuggingFace hub) to train on.",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data.",
    )
    parser.add_argument("--image_column", type=str, default="image", help="The column of the dataset containing the target image.")
    parser.add_argument("--conditioning_image_column", type=str, default="conditioning_image", help="The column of the dataset containing the conditioning image.")
    parser.add_argument("--caption_column", type=str, default="text", help="The column of the dataset containing a caption.")
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of training examples.",
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings.",
    )
    parser.add_argument(
        "--conditioning_image_drop_rate",
        type=float,
        default=0.0,
        help="Probability of dropping the conditioning image during training.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help="A set of prompts evaluated every `--validation_steps`.",
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help="A set of paths to the controlnet conditioning image.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each validation prompt.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="flux_controlnet_mobilenet",
        help="The `project_name` argument passed to Accelerator.init_trackers.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Guidance scale for FLUX.dev during validation.",
    )
    parser.add_argument(
        "--freeze_mobilenet",
        action="store_true",
        help="Freeze the MobileNetV3 backbone (keeps ImageNet features; trains only the LoRA projections and adapter convs).",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=32,
        help="Rank r for LoRA projections in MobileNet Flux ControlNet (default: 32).",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=None,
        help="Alpha scaling for LoRA projections (default: same as lora_rank).",
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


def get_train_dataset(args, accelerator):
    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=getattr(args, 'cache_dir', None),
            data_dir=args.train_data_dir,
        )
    else:
        if args.train_data_dir is not None:
            dataset_dir = args.train_data_dir
            if os.path.exists(os.path.join(dataset_dir, "dataset_info.json")) or os.path.exists(os.path.join(dataset_dir, "train")):
                dataset = load_from_disk(dataset_dir)
            else:
                dataset = load_dataset(
                    dataset_dir,
                    cache_dir=getattr(args, 'cache_dir', None),
                )

    column_names = dataset["train"].column_names

    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(f"`--image_column` value '{args.image_column}' not found in dataset columns.")

    if args.caption_column is None:
        caption_column = column_names[1]
        logger.info(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(f"`--caption_column` value '{args.caption_column}' not found in dataset columns.")

    if args.conditioning_image_column is None:
        conditioning_image_column = column_names[2]
        logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
    else:
        conditioning_image_column = args.conditioning_image_column
        if conditioning_image_column not in column_names:
            raise ValueError(f"`--conditioning_image_column` value '{args.conditioning_image_column}' not found in dataset columns.")

    with accelerator.main_process_first():
        train_dataset = dataset["train"].shuffle(seed=args.seed)
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples))
    return train_dataset


def encode_prompt(prompt_batch, text_encoder, text_encoder_2, tokenizer, tokenizer_2, proportion_empty_prompts, is_train=True):
    pooled_prompt_embeds = None
    prompt_embeds = None

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        clip_output = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )
        pooled_prompt_embeds = clip_output.pooler_output

        text_inputs_2 = tokenizer_2(
            captions,
            padding="max_length",
            max_length=tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids_2 = text_inputs_2.input_ids
        prompt_embeds = text_encoder_2(
            text_input_ids_2.to(text_encoder_2.device),
            output_hidden_states=False,
        )[0]

    return prompt_embeds, pooled_prompt_embeds


def prepare_train_dataset(dataset, accelerator, args):
    image_transforms = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    conditioning_image_transforms = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
    ])

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[args.image_column]]
        images = [image_transforms(image) for image in images]

        conditioning_images = [image.convert("RGB") for image in examples[args.conditioning_image_column]]
        conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]

        if args.conditioning_image_drop_rate > 0:
            conditioning_images = [
                torch.zeros_like(img) if random.random() < args.conditioning_image_drop_rate else img
                for img in conditioning_images
            ]

        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_images

        return examples

    with accelerator.main_process_first():
        dataset = dataset.with_transform(preprocess_train)

    return dataset


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    prompt_embeds = torch.stack([torch.tensor(example["prompt_embeds"]) for example in examples])
    pooled_prompt_embeds = torch.stack([torch.tensor(example["pooled_prompt_embeds"]) for example in examples])

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
    }


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `hf auth login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_2 = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    text_encoder_2 = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )

    from diffusers import FluxTransformer2DModel
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )

    if args.controlnet_model_name_or_path is not None:
        logger.info(f"Loading existing controlnet weights from {args.controlnet_model_name_or_path}")
        controlnet = MobileNetFluxControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        inner_dim = transformer.config["num_attention_heads"] * transformer.config["attention_head_dim"]
        controlnet = MobileNetFluxControlNetModel(
            flux_hidden_dim=inner_dim,
            guidance_embeds=transformer.config["guidance_embeds"],
            num_layers=19,
            num_single_layers=38,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha if args.lora_alpha is not None else args.lora_rank,
        )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1
                while len(weights) > 0:
                    weights.pop()
                    model = models[i]
                    sub_dir = "controlnet"
                    save_dir = os.path.join(output_dir, sub_dir)
                    os.makedirs(save_dir, exist_ok=True)
                    model.save_pretrained(save_dir)
                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                model = models.pop()
                weights_path = os.path.join(input_dir, "controlnet")
                loaded_model = MobileNetFluxControlNetModel.from_pretrained(weights_path)
                model.load_state_dict(loaded_model.state_dict())
                del loaded_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    controlnet.train()
    controlnet.requires_grad_(True)

    if args.freeze_mobilenet:
        if hasattr(controlnet.mobilenet_adapter, "stage1"):
            for stage in [
                controlnet.mobilenet_adapter.stage1,
                controlnet.mobilenet_adapter.stage2,
                controlnet.mobilenet_adapter.stage3,
                controlnet.mobilenet_adapter.stage4,
            ]:
                for p in stage.parameters():
                    p.requires_grad_(False)
        elif hasattr(controlnet.mobilenet_adapter, "features"):
            for p in controlnet.mobilenet_adapter.features.parameters():
                p.requires_grad_(False)
        total_params = sum(p.numel() for p in controlnet.parameters())
        trainable_params = sum(p.numel() for p in controlnet.parameters() if p.requires_grad)
        logger.info(
            f"MobileNetV3 backbone FROZEN: {trainable_params/1e6:.2f}M trainable / {total_params/1e6:.2f}M total params"
        )

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        controlnet.enable_gradient_checkpointing()

    transformer = torch.compile(transformer)

    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training."
    )
    if unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"ControlNet loaded as datatype {unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = controlnet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)

    def compute_embeddings(batch, proportion_empty_prompts, text_encoder, text_encoder_2, tokenizer, tokenizer_2, is_train=True):
        prompt_batch = batch[args.caption_column]
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            prompt_batch, text_encoder, text_encoder_2, tokenizer, tokenizer_2, proportion_empty_prompts, is_train
        )
        prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=weight_dtype)
        return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}

    train_dataset = get_train_dataset(args, accelerator)
    compute_embeddings_fn = functools.partial(
        compute_embeddings,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        proportion_empty_prompts=args.proportion_empty_prompts,
    )

    with accelerator.main_process_first():
        from datasets.fingerprint import Hasher
        new_fingerprint = Hasher.hash(args)
        train_dataset = train_dataset.map(compute_embeddings_fn, batched=True, batch_size=64, new_fingerprint=new_fingerprint)

    del text_encoder, text_encoder_2, tokenizer, tokenizer_2
    gc.collect()
    torch.cuda.empty_cache()

    train_dataset = prepare_train_dataset(train_dataset, accelerator, args)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created."
            )
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompt", None)
        tracker_config.pop("validation_image", None)
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    best_loss = float("inf")
    best_step = None
    running_loss = 0.0
    running_loss_count = 0

    image_logs = None
    if global_step == 0 and args.validation_prompt is not None:
        if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
            logger.info("Running initial validation (baseline, controlnet at zero)...")
            log_validation(
                vae=vae,
                transformer=transformer,
                controlnet=controlnet,
                args=args,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                step=global_step,
            )

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
                    latents = latents.to(dtype=weight_dtype)

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Flow matching time-shift schedule (Recommendation 3)
                u = torch.rand(bsz, device=latents.device, dtype=latents.dtype)
                if args.flow_shift != 1.0:
                    shift = args.flow_shift
                    sigmas = (u * shift) / (1.0 + (shift - 1.0) * u)
                else:
                    sigmas = u

                while sigmas.ndim < latents.ndim:
                    sigmas = sigmas.unsqueeze(-1)

                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
                target = noise - latents

                conditioning_pixel_values = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
                latent_h = latents.shape[2]
                latent_w = latents.shape[3]

                prompt_embeds = batch["prompt_embeds"].to(dtype=weight_dtype)
                pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(dtype=weight_dtype)

                img_ids = FluxControlNetPipeline._prepare_latent_image_ids(
                    bsz, latent_h // 2, latent_w // 2, device=latents.device, dtype=latents.dtype
                )

                txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=latents.device, dtype=latents.dtype)

                timesteps = sigmas.reshape(bsz) * 1000.0

                guidance = torch.full((bsz,), args.guidance_scale, device=latents.device, dtype=latents.dtype)

                noisy_latents_permuted = FluxControlNetPipeline._pack_latents(
                    noisy_latents, bsz, noisy_latents.shape[1], latent_h, latent_w
                ).to(dtype=weight_dtype)

                controlnet_output = controlnet(
                    hidden_states=noisy_latents_permuted,
                    controlnet_cond=conditioning_pixel_values,
                    timestep=timesteps / 1000,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=guidance,
                    return_dict=False,
                )

                controlnet_block_samples = tuple(o.to(dtype=weight_dtype) for o in controlnet_output[0])
                controlnet_single_block_samples = tuple(o.to(dtype=weight_dtype) for o in controlnet_output[1])

                img = transformer(
                    hidden_states=noisy_latents_permuted,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    timestep=timesteps / 1000,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    guidance=guidance,
                    controlnet_block_samples=controlnet_block_samples,
                    controlnet_single_block_samples=controlnet_single_block_samples,
                ).sample

                model_pred = img

                target_packed = FluxControlNetPipeline._pack_latents(
                    target, bsz, target.shape[1], latent_h, latent_w
                ).to(dtype=weight_dtype)

                loss = F.mse_loss(model_pred.float(), target_packed.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), args.max_grad_norm)
                cn_grad_norm_val = _grad_norm(controlnet)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                running_loss += loss.detach().item()
                running_loss_count += 1

                if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                        # Evaluate average loss over the checkpoint interval
                        avg_interval_loss = running_loss / max(running_loss_count, 1)
                        running_loss = 0.0
                        running_loss_count = 0

                        logger.info(
                            f"[Step {global_step}] Interval average loss: {avg_interval_loss:.5f} (Previous best: {best_loss:.5f})"
                        )

                        # Save best model if this checkpoint sets a new low (after warmup)
                        if global_step > args.lr_warmup_steps and avg_interval_loss < best_loss:
                            best_loss = avg_interval_loss
                            best_step = global_step
                            best_dir = os.path.join(args.output_dir, "controlnet_best")
                            os.makedirs(best_dir, exist_ok=True)
                            unwrap_model(controlnet).save_pretrained(best_dir)
                            metric_path = os.path.join(args.output_dir, "best_metric.json")
                            with open(metric_path, "w") as f:
                                json.dump({"best_step": best_step, "best_loss": best_loss}, f, indent=2)
                            logger.info(
                                f"*** New best ControlNet saved at step {best_step} (loss: {best_loss:.5f}) in {best_dir} ***"
                            )

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        image_logs = log_validation(
                            vae=vae,
                            transformer=transformer,
                            controlnet=controlnet,
                            args=args,
                            accelerator=accelerator,
                            weight_dtype=weight_dtype,
                            step=global_step,
                        )

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "best_loss": best_loss if best_loss != float("inf") else loss.detach().item(),
                "cn_grad_norm": cn_grad_norm_val,
                "cn_res_norm": controlnet_block_samples[0].detach().float().norm().item(),
                "cn_loraB_norm": _weight_norm(controlnet.controlnet_blocks[0].lora_B),
                "cn_loraA_norm": _weight_norm(controlnet.controlnet_blocks[0].lora_A),
                "cn_convout_norm": _weight_norm(controlnet.mobilenet_adapter.conv_out.weight),
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet_model = unwrap_model(controlnet)
        save_dir = os.path.join(args.output_dir, "controlnet")
        controlnet_model.save_pretrained(save_dir)

        # Fallback if no best checkpoint was saved during training
        best_dir = os.path.join(args.output_dir, "controlnet_best")
        if not os.path.exists(best_dir):
            controlnet_model.save_pretrained(best_dir)
            metric_path = os.path.join(args.output_dir, "best_metric.json")
            with open(metric_path, "w") as f:
                json.dump({"best_step": global_step, "best_loss": loss.detach().item()}, f, indent=2)

        image_logs = None
        if args.validation_prompt is not None:
            image_logs = log_validation(
                vae=vae,
                transformer=transformer,
                controlnet=controlnet,
                args=args,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                step=global_step,
                is_final_validation=True,
            )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
