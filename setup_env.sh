#!/usr/bin/env bash
# ==============================================================================
# setup_env.sh - PERSON_EXTRACTOR_V4 environment setup (Linux/Mac)
# ==============================================================================
# Creates an isolated virtual environment, installs all dependencies,
# downloads the spaCy model, generates synthetic training data, trains
# the LightGBM classifier, and runs the test suite.
#
# Usage:
#     bash setup_env.sh
# ==============================================================================
set -e

step() {
    echo ""
    echo "==> $1"
}

step "Creating virtual environment (.venv)"
if [ -d ".venv" ]; then
    echo "  .venv already exists, skipping creation."
else
    python3 -m venv .venv
fi

step "Activating virtual environment"
source .venv/bin/activate

step "Upgrading pip"
python -m pip install --upgrade pip

step "Installing requirements"
python -m pip install -r requirements.txt

step "Downloading spaCy English model (en_core_web_sm)"
python -m spacy download en_core_web_sm

step "Generating synthetic training data"
python scripts/generate_training_data.py

step "Training the LightGBM classifier"
python scripts/train_model.py

step "Running the test suite"
python -m pytest tests/ -v

echo ""
echo "============================================================="
echo " SETUP COMPLETE"
echo "============================================================="
echo ""
echo "Try it now:"
echo "  python cli.py --input path/to/your/file.txt --explain"
echo "  python cli.py --evaluate"
echo ""
echo "To improve results on YOUR documents over time:"
echo "  python scripts/label_feedback.py        (confirm/correct REVIEW items)"
echo "  python scripts/retrain_from_feedback.py  (fold corrections into the model)"
echo ""
echo "Remember to activate the environment in future sessions:"
echo "  source .venv/bin/activate"
echo ""
