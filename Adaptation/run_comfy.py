#!/usr/bin/env python3
"""
ComfyUI workflow runner for domain adaptation.

Processes paired images (images, depth, conditioning_images) through the Flux workflow.

Usage:
    # Local (ComfyUI must be running already):
    python run_comfy.py

    # Local (ComfyUI auto-start):
    python run_comfy.py --auto-start

    # Slurm: sbatch run_comfy.slurm [--start N --end M]

    # Process a subset:
    python run_comfy.py --start 10 --end 19

    # Only setup inputs, then exit:
    python run_comfy.py --setup-only
"""

import argparse
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

log = logging.getLogger("comfy_runner")


# ─── Paths ───────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
WORKFLOW_PATH = Path(os.environ.get("WORKFLOW_PATH", BASE_DIR / "Workflow_final.json"))

def get_dataset_dir() -> Path:
    if "DATASET_DIR" in os.environ:
        return Path(os.environ["DATASET_DIR"])
    for candidate in ("Dataset", "dataset_synthia_seg", "dataset_comfy"):
        p = BASE_DIR / candidate
        if p.exists() and (p / "images").exists():
            return p
    return BASE_DIR / "Dataset"

DATASET_DIR = get_dataset_dir()
OUTPUT_DIR = BASE_DIR / "output"

COMFYUI_DIR = Path(os.environ.get("COMFYUI_DIR", Path.home() / "ComfyUI"))
COMFYUI_HOST = os.environ.get("COMFYUI_HOST", "127.0.0.1")
COMFYUI_PORT = int(os.environ.get("COMFYUI_PORT", "8188"))
API_BASE = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"

COMFYUI_INPUT = COMFYUI_DIR / "input"
COMFYUI_OUTPUT = COMFYUI_DIR / "output"

INPUT_PREFIX = "domain_adaptation"

# Node id -> dataset subdir
INPUT_NODES = {
    "1": "images",
    "121:28": "Depth",
    "121:135": "conditioning_images",
    "121:145": "mask",
}

OUTPUT_NODE = "64"

# ComfyUI keeps the history of every executed prompt in RAM; on long API
# batches this grows until the cgroup memory limit is hit (OOM kill after
# ~600 images in our runs). Clear it periodically.
HISTORY_CLEAR_INTERVAL = 50


def get_image_files() -> List[str]:
    img_dir = DATASET_DIR / "images"
    if not img_dir.exists():
        return []
    pattern = re.compile(r"^.+\.(png|jpg|jpeg|webp)$", re.IGNORECASE)
    return sorted([f.name for f in img_dir.iterdir() if f.is_file() and pattern.match(f.name)])


def get_num_images() -> int:
    return len(get_image_files())


# ─── API ─────────────────────────────────────────────────────────────────────

def api_get(endpoint: str) -> Optional[dict]:
    url = f"{API_BASE}{endpoint}"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, OSError):
        return None


def api_post(endpoint: str, data: dict) -> Optional[dict]:
    url = f"{API_BASE}{endpoint}"
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
        log.error("API %s: %s", endpoint, getattr(e, "reason", e))
        return None


def health_check() -> bool:
    return api_get("/system_stats") is not None


def clear_history() -> bool:
    """POST /history {"clear": true}; response body is empty."""
    url = f"{API_BASE}/history"
    body = json.dumps({"clear": True}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            r.read()
            return True
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
        log.error("clear_history: %s", getattr(e, "reason", e))
        return False


def log_job_ram() -> Optional[float]:
    """Total RAM used by this job's cgroup, in GB (diagnostic)."""
    try:
        rel = open("/proc/self/cgroup").read().strip().split(":")[-1]
        if rel == "/":
            return None
        sub = rel.lstrip("/")
        for name in ("memory.current", "memory.usage_in_bytes"):
            p = Path("/sys/fs/cgroup") / sub / name
            if p.exists():
                return int(p.read_text().strip()) / (1024**3)
    except Exception:
        pass
    return None


# ─── Setup ───────────────────────────────────────────────────────────────────

def setup_inputs(image_files: Optional[List[str]] = None):
    target = COMFYUI_INPUT / INPUT_PREFIX
    target.mkdir(parents=True, exist_ok=True)
    if image_files is None:
        image_files = get_image_files()
    count = 0
    for node_id, subdir in INPUT_NODES.items():
        src_dir = DATASET_DIR / subdir
        dst_dir = target / subdir
        dst_dir.mkdir(parents=True, exist_ok=True)
        for fname in image_files:
            src = src_dir / fname
            dst = dst_dir / fname
            if not src.exists():
                log.warning("Missing source: %s", src)
                continue
            if not dst.exists():
                shutil.copy2(src, dst)
                count += 1
    log.info("Prepared inputs in %s (%d new files copied)", target, count)


# ─── Workflow ────────────────────────────────────────────────────────────────

def convert_ui_to_api(ui_wf: dict) -> dict:
    if "nodes" not in ui_wf or not isinstance(ui_wf["nodes"], list):
        return ui_wf
    
    links = {l[0]: (str(l[1]), l[2]) for l in ui_wf.get("links", [])}
    prompt = {}
    
    for node in ui_wf["nodes"]:
        nid = str(node["id"])
        ctype = node["type"]
        inputs = {}
        
        for inp in node.get("inputs", []):
            iname = inp["name"]
            lid = inp.get("link")
            if lid is not None and lid in links:
                from_node, from_slot = links[lid]
                inputs[iname] = [from_node, from_slot]
                
        wv = node.get("widgets_values", [])
        if isinstance(wv, list):
            if ctype == "LoadImage":
                if len(wv) > 0: inputs["image"] = wv[0]
                if len(wv) > 1: inputs["upload"] = wv[1]
            elif ctype == "DualCLIPLoader":
                if len(wv) > 0: inputs["clip_name1"] = wv[0]
                if len(wv) > 1: inputs["clip_name2"] = wv[1]
                if len(wv) > 2: inputs["type"] = wv[2]
                if len(wv) > 3: inputs["device"] = wv[3]
            elif ctype == "VAELoader":
                if len(wv) > 0: inputs["vae_name"] = wv[0]
            elif ctype == "UNETLoader":
                if len(wv) > 0: inputs["unet_name"] = wv[0]
                if len(wv) > 1: inputs["weight_dtype"] = wv[1]
            elif ctype == "SaveImage":
                if len(wv) > 0: inputs["filename_prefix"] = wv[0]
            elif ctype == "ControlNetLoader":
                if len(wv) > 0: inputs["control_net_name"] = wv[0]
            elif ctype == "MobileNetFluxControlNetLoader":
                if len(wv) > 0: inputs["model_name"] = wv[0]
            elif ctype == "LoraLoaderModelOnly":
                if len(wv) > 0: inputs["lora_name"] = wv[0]
                if len(wv) > 1: inputs["strength_model"] = wv[1]
            elif ctype == "CLIPTextEncode":
                if len(wv) > 0: inputs["text"] = wv[0]
            elif ctype == "FluxGuidance":
                if len(wv) > 0: inputs["guidance"] = wv[0]
            elif ctype == "ModelSamplingFlux":
                if len(wv) > 0: inputs["max_shift"] = wv[0]
                if len(wv) > 1: inputs["base_shift"] = wv[1]
                if len(wv) > 2: inputs["width"] = wv[2]
                if len(wv) > 3: inputs["height"] = wv[3]
            elif ctype == "ControlNetApplyAdvanced":
                if len(wv) > 0: inputs["strength"] = wv[0]
                if len(wv) > 1: inputs["start_percent"] = wv[1]
                if len(wv) > 2: inputs["end_percent"] = wv[2]
            elif ctype == "Image To Mask":
                if len(wv) > 0: inputs["method"] = wv[0]
            elif ctype == "InpaintModelConditioning":
                if len(wv) > 0: inputs["noise_mask"] = wv[0]
            elif ctype == "KSampler":
                if len(wv) > 0: inputs["seed"] = wv[0]
                if len(wv) > 2: inputs["steps"] = wv[2]
                if len(wv) > 3: inputs["cfg"] = wv[3]
                if len(wv) > 4: inputs["sampler_name"] = wv[4]
                if len(wv) > 5: inputs["scheduler"] = wv[5]
                if len(wv) > 6: inputs["denoise"] = wv[6]
            elif ctype == "GrowMask":
                if len(wv) > 0: inputs["expand"] = wv[0]
                if len(wv) > 1: inputs["tapered_corners"] = wv[1]
            elif ctype == "MaskToSEGS":
                if len(wv) > 0: inputs["combined"] = wv[0]
                if len(wv) > 1: inputs["crop_factor"] = wv[1]
                if len(wv) > 2: inputs["bbox_fill"] = wv[2]
                if len(wv) > 3: inputs["drop_size"] = wv[3]
                if len(wv) > 4: inputs["contour_fill"] = wv[4]
            elif ctype == "DetailerForEach":
                if len(wv) > 0: inputs["guide_size"] = wv[0]
                if len(wv) > 1: inputs["guide_size_for"] = wv[1]
                if len(wv) > 2: inputs["max_size"] = wv[2]
                if len(wv) > 3: inputs["seed"] = wv[3]
                if len(wv) > 5: inputs["steps"] = wv[5]
                if len(wv) > 6: inputs["cfg"] = wv[6]
                if len(wv) > 7: inputs["sampler_name"] = wv[7]
                if len(wv) > 8: inputs["scheduler"] = wv[8]
                if len(wv) > 9: inputs["denoise"] = wv[9]
                if len(wv) > 10: inputs["feather"] = wv[10]
                if len(wv) > 11: inputs["noise_mask"] = wv[11]
                if len(wv) > 12: inputs["force_inpaint"] = wv[12]
                if len(wv) > 13: inputs["wildcard"] = wv[13]
                if len(wv) > 14: inputs["cycle"] = wv[14]
                if len(wv) > 15: inputs["inpaint_model"] = wv[15]
                if len(wv) > 16: inputs["noise_mask_feather"] = wv[16]
                if len(wv) > 17: inputs["tiled_encode"] = wv[17]
                if len(wv) > 18: inputs["tiled_decode"] = wv[18]
                
        prompt[nid] = {
            "inputs": inputs,
            "class_type": ctype,
            "_meta": {"title": node.get("title") or ctype}
        }
        
    return prompt


def load_workflow() -> dict:
    if not WORKFLOW_PATH.exists():
        raise FileNotFoundError(f"Workflow file not found: {WORKFLOW_PATH}")
    with open(WORKFLOW_PATH) as f:
        data = json.load(f)
    if isinstance(data, dict) and "nodes" in data:
        log.info("Converting UI workflow format to API prompt format")
        return convert_ui_to_api(data)
    return data


def set_image(workflow: dict, node_id: str, fname: str):
    if node_id not in workflow:
        log.warning("Node %s not found in workflow", node_id)
        return
    subdir = INPUT_NODES[node_id]
    workflow[node_id]["inputs"]["image"] = f"{INPUT_PREFIX}/{subdir}/{fname}"


def randomize_seeds(workflow: dict):
    for node_id, node in workflow.items():
        if isinstance(node, dict) and "inputs" in node:
            inputs = node["inputs"]
            for k in ("seed", "noise_seed"):
                if k in inputs and isinstance(inputs[k], int):
                    inputs[k] = random.randint(0, 2**32 - 1)


# ─── Prompt ──────────────────────────────────────────────────────────────────

def queue_prompt(workflow: dict) -> Optional[str]:
    resp = api_post("/prompt", {"prompt": workflow})
    if resp:
        return resp.get("prompt_id")
    return None


def wait_prompt(prompt_id: str, timeout: int = 600) -> Optional[dict]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        history = api_get(f"/history/{prompt_id}")
        if history and prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("completed"):
                messages = status.get("messages", [])
                if any(m[0] == "execution_error" for m in messages):
                    log.error("Prompt %s failed", prompt_id)
                    return None
                return entry.get("outputs", {})
        time.sleep(3)
    log.error("Timeout waiting for prompt %s", prompt_id)
    return None


def copy_outputs(index: int, outputs: dict, fname: Optional[str] = None):
    base_name = Path(fname).stem if fname else f"{index:05d}"
    target_outputs = {OUTPUT_NODE: outputs[OUTPUT_NODE]} if OUTPUT_NODE in outputs else outputs
    for node_id, node_out in target_outputs.items():
        for img in node_out.get("images", []):
            src = COMFYUI_OUTPUT / img.get("subfolder", "") / img["filename"]
            if src.exists():
                dst = OUTPUT_DIR / f"{base_name}_{img['filename']}"
                shutil.copy2(src, dst)
                log.info("Saved: %s", dst.name)
            else:
                log.warning("Output file not found: %s", src)


# ─── ComfyUI lifecycle ───────────────────────────────────────────────────────

def start_comfyui() -> Optional[subprocess.Popen]:
    main_py = COMFYUI_DIR / "main.py"
    if not main_py.exists():
        log.error("ComfyUI main.py not found at %s", main_py)
        return None
    log.info("Starting ComfyUI...")
    log_path = Path(os.environ.get("SLURM_LOGS_DIR", "/tmp")) / f"comfyui_{os.environ.get('SLURM_JOB_ID', 'local')}.log"
    with open(log_path, "w") as lf:
        proc = subprocess.Popen(
            [sys.executable, str(main_py), "--listen", COMFYUI_HOST, "--port", str(COMFYUI_PORT)],
            stdout=lf,
            stderr=lf,
        )
    for _ in range(60):
        if health_check():
            log.info("ComfyUI is ready")
            return proc
        time.sleep(2)
    log.error("ComfyUI failed to start")
    proc.kill()
    return None


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run ComfyUI workflow for domain adaptation"
    )
    parser.add_argument("--start", type=int, default=None, help="First image index")
    parser.add_argument("--end", type=int, default=None, help="Last image index (inclusive, default: auto-detect)")
    parser.add_argument("--num-chunks", type=int, default=None, help="Total number of chunks to split dataset into")
    parser.add_argument("--chunk-idx", type=int, default=None, help="Index of chunk to process (0-based)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save output images")
    parser.add_argument("--setup-only", action="store_true", help="Only prepare files in ComfyUI input")
    parser.add_argument("--auto-start", action="store_true", help="Start ComfyUI automatically")
    parser.add_argument("--clear-history-every", type=int, default=HISTORY_CLEAR_INTERVAL,
                        help="Clear ComfyUI history every N images to bound RAM usage (0 = never)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)

    image_files = get_image_files()
    n = len(image_files)
    if n == 0:
        log.error("No images found in %s", DATASET_DIR / "images")
        sys.exit(1)

    # Determine start and end indices
    start = args.start
    end = args.end
    num_chunks = args.num_chunks
    chunk_idx = args.chunk_idx

    # If start/end/num-chunks not explicitly given, check Slurm array environment variables
    if start is None and end is None and num_chunks is None:
        if "SLURM_ARRAY_TASK_ID" in os.environ:
            chunk_idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
            if "SLURM_ARRAY_TASK_COUNT" in os.environ:
                num_chunks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
            elif "SLURM_ARRAY_TASK_MAX" in os.environ and "SLURM_ARRAY_TASK_MIN" in os.environ:
                num_chunks = int(os.environ["SLURM_ARRAY_TASK_MAX"]) - int(os.environ["SLURM_ARRAY_TASK_MIN"]) + 1

    if num_chunks is not None and num_chunks > 0:
        if chunk_idx is None:
            chunk_idx = 0
        chunk_size = (n + num_chunks - 1) // num_chunks
        start = chunk_idx * chunk_size
        end = min(n - 1, (chunk_idx + 1) * chunk_size - 1)
        log.info("Chunk %d/%d assigned range: [%d, %d] (chunk size: %d, total: %d)",
                 chunk_idx, num_chunks, start, end, chunk_size, n)

    if start is None:
        start = 0
    if end is None:
        end = n - 1
    else:
        end = min(end, n - 1)

    if start < 0 or start >= n or start > end:
        log.error("Invalid range: start=%d, end=%d (total images: %d)", start, end, n)
        sys.exit(1)

    selected_files = image_files[start : end + 1]

    # ── Setup inputs ──
    setup_inputs(image_files if args.setup_only else selected_files)
    if args.setup_only:
        return

    # ── Output dir ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Start / wait for ComfyUI ──
    proc = None
    if args.auto_start:
        proc = start_comfyui()
        if proc is None:
            log.error("Could not start ComfyUI")
            sys.exit(1)
    elif not health_check():
        log.error("ComfyUI not reachable at %s", API_BASE)
        log.error("Start ComfyUI manually or use --auto-start")
        sys.exit(1)

    # ── Load workflow ──
    workflow = load_workflow()

    # ── Process images ──
    ok = fail = 0
    total = end - start + 1
    for idx in range(start, end + 1):
        fname = image_files[idx]
        try:
            for node_id in INPUT_NODES:
                set_image(workflow, node_id, fname)
            randomize_seeds(workflow)
            log.info("[%d/%d] Processing %s (idx %05d)", idx - start + 1, total, fname, idx)
            pid = queue_prompt(workflow)
            if not pid:
                if not health_check():
                    log.error("ComfyUI is unreachable, aborting")
                    break
                fail += 1
                continue
            outputs = wait_prompt(pid)
            if outputs is not None:
                copy_outputs(idx, outputs, fname)
                ok += 1
            else:
                if not health_check():
                    log.error("ComfyUI is unreachable, aborting")
                    break
                fail += 1
            if args.clear_history_every > 0 and ok + fail > 0 and (ok + fail) % args.clear_history_every == 0:
                if clear_history():
                    log.info("Cleared ComfyUI history (limit RAM growth)")
                ram = log_job_ram()
                if ram is not None:
                    log.info("Job RAM: %.2f GB", ram)
        except Exception:
            log.exception("Error at index %d (%s)", idx, fname)
            if not health_check():
                log.error("ComfyUI is unreachable, aborting")
                break
            fail += 1

    log.info("Done: %d OK, %d FAILED (out of %d)", ok, fail, total)

    if proc:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    main()
