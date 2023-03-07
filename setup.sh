#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd "$SCRIPT_DIR"

if [ ! -d ./.venv ]; then
	python -m venv .venv
	echo "Created .venv"
fi

echo "Activating .venv"
source .venv/bin/activate

echo "Installing requirements..."
pip install -r requirements.txt

echo "Done! REMEMBER TO ACTIVATE VENV with 'source .venv/bin/activate'"

