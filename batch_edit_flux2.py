from __future__ import annotations

import math
import numpy as np
from PIL import Image
import argparse
import gc
import sys
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image, ImageOps
from diffusers import Flux2KleinPipeline



def round_up_to_multiple(value: int, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


def pad_image_to_multiple(im: Image.Image, multiple: int = 16):
    """
    Pads image out to the nearest multiple using edge padding.
    Returns:
        padded_image, crop_box
    where crop_box = (left, top, right, bottom) for cropping back later.
    """
    w, h = im.size
    new_w = round_up_to_multiple(w, multiple)
    new_h = round_up_to_multiple(h, multiple)

    if new_w == w and new_h == h:
        return im, (0, 0, w, h)

    pad_w = new_w - w
    pad_h = new_h - h

    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    arr = np.array(im)
    arr = np.pad(
        arr,
        ((top, bottom), (left, right), (0, 0)),
        mode="edge"
    )

    padded = Image.fromarray(arr)
    crop_box = (left, top, left + w, top + h)
    return padded, crop_box

def resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        if not args.prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")

        prompt_text = args.prompt_file.read_text(encoding="utf-8").strip()
        if not prompt_text:
            raise ValueError(f"Prompt file is empty: {args.prompt_file}")

        return prompt_text

    if args.prompt:
        return args.prompt.strip()

    raise ValueError("No prompt provided.")

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch image editing with FLUX.2 [klein] models."
    )
    parser.add_argument("--input-dir", required=True, type=Path, help="Folder of source images.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder for edited images.")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Edit instruction applied to every image.",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Path to a text file containing the edit prompt.",
    )
    parser.add_argument(
        "--model",
        default="black-forest-labs/FLUX.2-klein-4B",
        help="HF model id. Examples: black-forest-labs/FLUX.2-klein-9B, "
             "black-forest-labs/FLUX.2-klein-9b-kv, black-forest-labs/FLUX.2-klein-4B",
    )
    parser.add_argument("--steps", type=int, default=4, help="Denoising steps.")
    parser.add_argument("--guidance", type=float, default=1.0, help="Guidance scale.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16"),
        default="bf16",
        help="Torch dtype to load the model with.",
    )
    parser.add_argument("--max-width", type=int, default=None, help="Optional max working width before downscaling.")
    parser.add_argument("--max-height", type=int, default=None, help="Optional max working height before downscaling.")

    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device. Usually 'cuda'.",
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=None,
        help="Optional fixed output width. If omitted, uses each input image width.",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=None,
        help="Optional fixed output height. If omitted, uses each input image height.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subfolders for images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    parser.add_argument(
        "--save-format",
        choices=("png", "jpg", "webp"),
        default="png",
        help="Output image format.",
    )
    parser.add_argument(
        "--filename-suffix",
        default="_edited",
        help="Suffix appended to each output filename.",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Enable model CPU offload to reduce VRAM use.",
    )
    parser.add_argument(
        "--pad-to-multiple",
        type=int,
        default=16,
        help="Pad images to a multiple before inference, then crop back to original size. Set 0 to disable.",
    )
    args = parser.parse_args()

    if not args.prompt and not args.prompt_file:
        parser.error("You must provide either --prompt or --prompt-file.")

    return args


def iter_images(folder: Path, recursive: bool):
    walker = folder.rglob("*") if recursive else folder.glob("*")
    for path in sorted(walker):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            yield path
   

def build_pipe(model_id: str, dtype_name: str, device: str, cpu_offload: bool) -> Flux2KleinPipeline:
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16

    pipe = Flux2KleinPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )

    # FLUX/Flux2 optimization hooks live on the VAE
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    return pipe


def output_path_for(src: Path, output_dir: Path, suffix: str, save_format: str) -> Path:
    return output_dir / f"{src.stem}{suffix}.{save_format}"


def main() -> int:
    args = parse_args()
    prompt = resolve_prompt(args)

    if not args.input_dir.exists():
        print(f"Input directory does not exist: {args.input_dir}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths_iter = iter_images(args.input_dir, args.recursive)
    first_path = next(paths_iter, None)
    if first_path is None:
        print("No supported images found.", file=sys.stderr)
        return 1

    print(f"Loading model: {args.model}")
    pipe = build_pipe(args.model, args.dtype, args.device, args.cpu_offload)

    for index, image_path in enumerate([first_path, *paths_iter]):  
        dst = output_path_for(image_path, args.output_dir, args.filename_suffix, args.save_format)
        if dst.exists() and not args.overwrite:
            print(f"[skip] {image_path.name} -> {dst.name} already exists")
            continue

        try:
            with Image.open(image_path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")

                if args.max_width or args.max_height:
                    max_w = args.max_width or im.width
                    max_h = args.max_height or im.height

                    scale = min(max_w / im.width, max_h / im.height, 1.0)
                    if scale < 1.0:
                        new_w = int(im.width * scale)
                        new_h = int(im.height * scale)
                        im = im.resize((new_w, new_h), Image.LANCZOS)
                        
                if args.target_width or args.target_height:
                    work_im = im
                    width = args.target_width or im.width
                    height = args.target_height or im.height
                    crop_box = None
                elif args.pad_to_multiple and args.pad_to_multiple > 0:
                    work_im, crop_box = pad_image_to_multiple(im, multiple=args.pad_to_multiple)
                    width, height = work_im.size
                else:
                    work_im = im
                    width, height = im.size
                    crop_box = None

                generator = torch.Generator(device="cpu").manual_seed(args.seed + index)

                result = pipe(
                    image=work_im,
                    prompt=prompt,
                    width=width,
                    height=height,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    generator=generator,
                ).images[0]

                if crop_box is not None:
                    result = result.crop(crop_box)

                save_kwargs = {"quality": 95} if args.save_format == "jpg" else {}
                result.save(dst, **save_kwargs)
                print(f"[ok] {image_path.name} -> {dst.name} ({result.width}x{result.height})")
        except Exception as exc:
            print(f"[error] {image_path.name}: {exc}", file=sys.stderr)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())