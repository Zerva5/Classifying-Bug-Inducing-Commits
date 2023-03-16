#!/bin/bash

# Change to the parent directory and then to the 'clones' directory
cd ../clones

# Initialize an empty file for the results
output_file="../data/all_apache_commits.csv"
echo "repo,sha,files_changed,diff_line_count" > "$output_file"

# Loop through all the subdirectories
for repo in */; do
	if [[ "$repo" == "apache-activemq/" || "$repo" == "apache-cassandra/" || "$repo" == "apache-groovy/" || "$repo" == "apache-hbase/" || "$repo" == "apache-ignite/" || "$repo" == "apache-spark/" || "$repo" == "apache-zookeeper/" || "$repo" == "apache-camel/" || "$repo" == "apache-flink/" || "$repo" == "apache-hadoop/" || "$repo" == "apache-hive/" || "$repo" == "apache-kafka/" || "$repo" == "apache-zeppelin/" ]]; then
		continue
	fi

	echo "Processing $repo"
	cd "$repo"

	# Get the list of all commits
	commits=$(git rev-list --all)

	function process_commit() {
		repo=$1
		commit=$2
		output_file=$3

		java_changes=$(git show --pretty="" --name-only "$commit" | grep '\.java$' | wc -l)

		# Check if the number of Java files changed is between 1 and 32
		if [ "$java_changes" -ge 1 ] && [ "$java_changes" -le 32 ]; then
			# Get the total diff lines for the Java files
			total_diff_lines=$(git show "$commit" -- "*.java" | grep '^[-+][^-+]' | wc -l)

			if [ "$total_diff_lines" -lt 250 ]; then
				# Save the result for this commit
				echo "${repo%/},$commit,$java_changes,$total_diff_lines" >> "../$output_file"
			fi
		fi
	}


	# Loop through the commits and count the number of Java files changed
	N=32
	while IFS= read -r commit; do
		((i=i%N)); ((i++==0)) && wait
		process_commit "$repo" "$commit" "$output_file" &
	done <<< "$commits"
	# Move back to the parent directory (clones)
	cd ..
done
