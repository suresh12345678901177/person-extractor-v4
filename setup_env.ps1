<#
==============================================================================
 setup_env.ps1  -  PERSON_EXTRACTOR_V4 environment setup (Windows)
==============================================================================
 Creates an isolated virtual environment, installs all dependencies,
 downloads the spaCy model, generates synthetic training data, trains
 the LightGBM classifier, and runs the test suite - a fresh checkout
 should be fully working after this one script.

 Usage (from the project root, in PowerShell):
     .\setup_env.ps1

 If script execution is blocked, run once first:
     Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
==============================================================================
#>

$ErrorActionPreference = "Stop"

function Step($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

Step "Creating virtual environment (.venv)"
if (Test-Path ".venv") {
    Write-Host "  .venv already exists, skipping creation." -ForegroundColor Yellow
} else {
    python -m venv .venv
}

Step "Activating virtual environment"
& .\.venv\Scripts\Activate.ps1

Step "Upgrading pip"
python -m pip install --upgrade pip

Step "Installing requirements"
python -m pip install -r requirements.txt

Step "Downloading spaCy English model (en_core_web_sm)"
python -m spacy download en_core_web_sm

Step "Generating synthetic training data"
python scripts\generate_training_data.py

Step "Training the LightGBM classifier"
python scripts\train_model.py

Step "Running the test suite"
python -m pytest tests\ -v

Write-Host ""
Write-Host "=============================================================" -ForegroundColor Green
Write-Host " SETUP COMPLETE" -ForegroundColor Green
Write-Host "=============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Try it now:" -ForegroundColor Yellow
Write-Host "  python cli.py --input path\to\your\file.txt --explain"
Write-Host "  python cli.py --evaluate"
Write-Host ""
Write-Host "To improve results on YOUR documents over time:" -ForegroundColor Yellow
Write-Host "  python scripts\label_feedback.py       (confirm/correct REVIEW items)"
Write-Host "  python scripts\retrain_from_feedback.py (fold corrections into the model)"
Write-Host ""
Write-Host "Remember to activate the environment in future sessions:" -ForegroundColor Yellow
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
