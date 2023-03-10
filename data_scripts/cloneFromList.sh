#!/bin/bash

# Check if a filename has been provided as an argument
if [ $# -eq 0 ]; then
  echo "Please provide a filename as an argument"
  exit 1
fi

# Read the filename from the first argument
filename=$1

# Check if the file exists
if [ ! -f "$filename" ]; then
  echo "$filename does not exist"
  exit 1
fi

# Create the 'repos' directory if it doesn't exist
if [ ! -d "commit_repos" ]; then
  mkdir repos
fi

# Read each line from the file and clone the corresponding repository
while read line; do
    # Extract the repository author and name from the URL
    repo_author=$(echo $line | awk -F: '{print $2}' | awk -F/ '{print $1}')
    repo_name=$(echo $line | awk -F/ '{print $NF}' | awk -F. '{print $1}')

    # Clone the repository into a directory with the author and name
    git clone $line "data/commit_repos/$repo_author:$repo_name"

done < $filename
