#!/bin/bash
cd ../clones

# Loop through all the subdirectories
for repo in */; do
  cd "$repo"

  # Get the list of 100 random commits
  commits=$(git rev-list --all)
  echo "$commits" | wc -l
  commits=$(echo "$commits" | shuf | head -n 100)

  # Initialize the total number of Java files changed
  total_java_changes=0
  commit_count_with_java_changes=0

  # Loop through the commits and count the number of Java files changed
  for commit in $commits; do
    java_changes=$(git show --pretty="" --name-only "$commit" | grep '\.java$' | wc -l)
    if [ "$java_changes" -gt 0 ]; then
      total_java_changes=$((total_java_changes + java_changes))
      commit_count_with_java_changes=$((commit_count_with_java_changes + 1))
    fi
  done

  # Calculate the average number of Java files changed per commit
  if [ "$commit_count_with_java_changes" -gt 0 ]; then
    average_java_changes=$(echo "scale=2; $total_java_changes / $commit_count_with_java_changes" | bc)
  else
    average_java_changes="0.00"
  fi

  # Save the result for this repository
  echo "${repo%/}, $average_java_changes, $commit_count_with_java_changes"

  # Move back to the parent directory (clones)
  cd ..
done
