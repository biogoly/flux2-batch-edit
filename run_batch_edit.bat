@echo off
setlocal

REM Move to the script directory
cd /d "%~dp0"

REM Activate the virtual environment
call .venv\Scripts\activate.bat

REM Create folders if they do not exist
if not exist "input" mkdir "input"
if not exist "output" mkdir "output"
if not exist "prompts" mkdir "prompts"

REM Run the batch edit script
python .\batch_edit_flux2.py ^
  --input-dir .\input ^
  --output-dir .\output ^
  --prompt-file .\prompts\edit_prompt.txt ^
  --model black-forest-labs/FLUX.2-klein-4b ^
  --steps 4 ^
  --guidance 1.0 ^
  --pad-to-multiple 16 ^
  --max-width 3840 ^
  --max-height 3840 ^
  --cpu-offload ^
  --overwrite

echo.
echo Done.
pause