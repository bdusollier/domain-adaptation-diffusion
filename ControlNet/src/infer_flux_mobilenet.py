import argparse
import json
import os
import sys
import torch
from PIL import Image
from torchvision import transforms

# Add project root to sys.path
PRJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PRJ_DIR not in sys.path:
    sys.path.insert(0, PRJ_DIR)

from diffusers import FluxControlNetPipeline
from transformers import AutoTokenizer, CLIPTextModel, T5EncoderModel
from src.flux.mobilenet_controlnet import MobileNetFluxControlNetModel


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with MobileNet FLUX ControlNet")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained FLUX.1-dev model directory",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        required=True,
        help="Path to trained ControlNet weights directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_results",
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--conditioning_images",
        type=str,
        nargs="+",
        default=[],
        help="Paths to conditioning/segmentation images",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=[],
        help="Prompts corresponding to the conditioning images",
    )
    parser.add_argument(
        "--gt_images",
        type=str,
        nargs="+",
        default=[],
        help="Optional paths to ground-truth images for side-by-side comparison",
    )
    parser.add_argument(
        "--metadata_json",
        type=str,
        default=None,
        help="Optional path to metadata.json to automatically pull test samples and prompts",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to generate if using metadata_json",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Image resolution for generation",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=20,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Guidance scale for FLUX",
    )
    parser.add_argument(
        "--controlnet_scale",
        type=float,
        default=1.0,
        help="Conditioning scale for ControlNet",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Precision format",
    )
    return parser.parse_args()


def create_comparison_grid(cond_img, gen_img, gt_img=None):
    width, height = gen_img.size
    cond_resized = cond_img.resize((width, height), Image.Resampling.NEAREST)
    
    if gt_img is not None:
        gt_resized = gt_img.resize((width, height), Image.Resampling.LANCZOS)
        grid = Image.new("RGB", (width * 3, height))
        grid.paste(cond_resized, (0, 0))
        grid.paste(gen_img, (width, 0))
        grid.paste(gt_resized, (width * 2, 0))
    else:
        grid = Image.new("RGB", (width * 2, height))
        grid.paste(cond_resized, (0, 0))
        grid.paste(gen_img, (width, 0))
    return grid


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dtype = torch.bfloat16 if args.mixed_precision == "bf16" else (
        torch.float16 if args.mixed_precision == "fp16" else torch.float32
    )

    # 1. Collect test pairs (cond, prompt, gt)
    samples = []
    if args.metadata_json and os.path.exists(args.metadata_json):
        base_dir = os.path.dirname(args.metadata_json)
        with open(args.metadata_json, "r") as f:
            meta = json.load(f)
        
        count = 0
        for item in meta:
            cond_rel = item.get("conditioning_image", item.get("conditioning_image_file_name", ""))
            img_rel = item.get("image", item.get("file_name", ""))
            prompt = item.get("prompt", item.get("text", "A realistic urban street scene."))
            
            cond_path = os.path.join(base_dir, cond_rel)
            if not os.path.exists(cond_path):
                cond_path = os.path.join(base_dir, "conditioning_images", os.path.basename(cond_rel))

            img_path = os.path.join(base_dir, img_rel)
            if not os.path.exists(img_path):
                img_path = os.path.join(base_dir, "images", os.path.basename(img_rel))

            if os.path.exists(cond_path):
                samples.append({
                    "cond": cond_path,
                    "prompt": prompt,
                    "gt": img_path if os.path.exists(img_path) else None,
                })
                count += 1
                if count >= args.num_samples:
                    break

    elif args.conditioning_images:
        for idx, cond_path in enumerate(args.conditioning_images):
            prompt = args.prompts[idx] if idx < len(args.prompts) else "A realistic urban street view, daytime driving perspective."
            gt = args.gt_images[idx] if idx < len(args.gt_images) else None
            samples.append({
                "cond": cond_path,
                "prompt": prompt,
                "gt": gt,
            })

    if not samples:
        print("ERROR: No valid conditioning samples found! Please provide --conditioning_images or --metadata_json.")
        sys.exit(1)

    print(f"=== Found {len(samples)} samples to generate ===")

    # 2. Load ControlNet
    print(f"Loading MobileNet ControlNet from: {args.controlnet_model_name_or_path}...")
    controlnet = MobileNetFluxControlNetModel.from_pretrained(
        args.controlnet_model_name_or_path,
        torch_dtype=dtype,
    )

    # 3. Load Pipeline
    print(f"Loading FluxControlNetPipeline from: {args.pretrained_model_name_or_path}...")
    pipeline = FluxControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=controlnet,
        torch_dtype=dtype,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=False)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Transform for conditioning image: exactly matches training [0, 1] range
    transform_pil = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(args.resolution),
    ])
    to_tensor = transforms.ToTensor()

    # 4. Generate images
    print(f"=== Starting inference on device: {device} ===")
    for idx, sample in enumerate(samples):
        print(f"\n--- [Sample {idx+1}/{len(samples)}] ---")
        print(f"Conditioning: {sample['cond']}")
        print(f"Prompt: {sample['prompt']}")

        raw_cond = Image.open(sample["cond"]).convert("RGB")
        cond_img = transform_pil(raw_cond)

        with torch.autocast(device_type=device, dtype=dtype):
            image = pipeline(
                prompt=sample["prompt"],
                control_image=cond_img,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                controlnet_conditioning_scale=args.controlnet_scale,
                generator=generator,
            ).images[0]

        # Save outputs
        gen_path = os.path.join(args.output_dir, f"sample_{idx+1:02d}_generated.png")
        image.save(gen_path)
        print(f"Saved generated image to: {gen_path}")

        gt_img = None
        if sample["gt"] and os.path.exists(sample["gt"]):
            raw_gt = Image.open(sample["gt"]).convert("RGB")
            gt_img = transform_pil(raw_gt)

        grid = create_comparison_grid(cond_img, image, gt_img)
        grid_path = os.path.join(args.output_dir, f"sample_{idx+1:02d}_comparison.png")
        grid.save(grid_path)
        print(f"Saved comparison grid (Mask | Generated | GroundTruth) to: {grid_path}")

    print(f"\n=== All {len(samples)} inference runs completed successfully! ===")
    print(f"Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
