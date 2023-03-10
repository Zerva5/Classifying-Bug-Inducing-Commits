#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cd "$SCRIPT_DIR"

if [ ! -d ./.venv ]; then
	python3 -m venv .venv
	echo "Created .venv"
fi

echo "Activating .venv"
source .venv/bin/activate

echo "Installing requirements..."
pip install -r requirements.txt

echo "Setting up 'data' directory"
echo "Creating 'data' directory if not already there"
mkdir data

mkdir data/datasets
mkdir data/commit_repos

echo "Cloning dataset repos..."
cd data
git clone git@github.com:hosseinkshvrz/apachejit.git
git clone git@github.com:grosa1/icse2021-szz-replication-package.git


echo "Done! REMEMBER TO ACTIVATE VENV with 'source .venv/bin/activate'"

