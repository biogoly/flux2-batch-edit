<img width="1280" height="602" alt="7b8a7873-e992-4646-9f6e-c3065800feaa" src="https://github.com/user-attachments/assets/9364c920-b7b4-4bbd-9bce-dd7c2e75538f" />


# FLUX.2 Klein Batch Image Editor

A Windows-friendly batch image editing tool for quickly processing a folder of images with FLUX.2 Klein using a shared edit prompt.

## Features

- Batch process all images in a folder
- Uses FLUX.2 Klein through Diffusers pipeline
- Prompt can be loaded from a text file
- Preserves final image dimensions by padding to a model-friendly multiple, then cropping back
- Optional max-width / max-height resizing to avoid VRAM issues
- Simple Windows `run_edit.bat` launcher

## Project Structure

```text
flux2-batch-edit/
├─ batch_edit_flux2.py
├─ run_edit.bat
├─ requirements.txt
├─ README.md
├─ .gitignore
├─ prompts/
│  └─ edit_prompt.txt
├─ input/
│  └─ .gitkeep
└─ output/
   └─ .gitkeep
```
## Requirements

```text
Windows
Python 3.12 recommended
NVIDIA GPU recommended
Hugging Face account with access to the selected FLUX.2 model
Setup
```

### Create and activate a virtual environment:
```
py -3.12 -m venv .venv
.\.venv\Scripts\activate
```
### Install PyTorch with CUDA support:
```
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
### Install dependencies:
```
python -m pip install -r requirements.txt
```
### Log into Hugging Face:
```
hf auth login
```
### Usage

Put source images into:
```
input/
```
Edit the prompt file:
```
prompts/edit_prompt.txt
```
Run: 
```
run_edit.bat
```
Edited images will be saved to:
```
output/
```
### Example Command
```
python .\batch_edit_flux2.py ^
  --input-dir .\input ^
  --output-dir .\output ^
  --prompt-file .\prompts\edit_prompt.txt ^
  --model black-forest-labs/FLUX.2-klein-4B ^
  --steps 4 ^
  --guidance 1.0 ^
  --pad-to-multiple 16 ^
  --max-width 1600 ^
  --max-height 1600 ^
  --overwrite
```
### Notes
Large images can cause CUDA out-of-memory errors. Use --max-width and --max-height to control working resolution.
The script pads images to a multiple of 16 before inference and crops back afterward.
PyTorch is installed separately because CUDA wheel selection depends on your system.
