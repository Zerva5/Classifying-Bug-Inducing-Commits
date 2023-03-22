import os
import pickle
import gc
from load_lookups import load_commit_lookup
from process_metadata import processNames, processTimestamps, processCommitMessages
from tqdm.auto import tqdm
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Commit:
    
    ###################################### SETUP ###################################### 
    
    def __init__(self, sha, repo_path, use_github = False):
        self.sha = sha
        self.repo_path = repo_path
        self.parent = None
        self.author = None
        self.datetime = None
        self.message = None
        self.bag_of_contexts = None
        self.use_github = use_github

#############################
POSITIVE_CSV_FILE = "./apache_positive_pairs.csv"
NEGATIVE_CSV_FILE = "./apache_negative_pairs.csv"
pickle_dir = '../data/commit_lookups'
output_file = '../data/commit_lookups/priority_commit_lookups3.pickle'
#############################

global COMMIT_LOOKUP
COMMIT_LOOKUP = {}

if not os.path.exists(os.path.dirname(output_file)):
    os.makedirs(os.path.dirname(output_file))

# Load commit pairs from the CSV files
positive_pairs = pd.read_csv(POSITIVE_CSV_FILE)
negative_pairs = pd.read_csv(NEGATIVE_CSV_FILE)

# Combine fix_hash and bug_hash from both files
relevant_shas = set(positive_pairs['fix_hash']).union(positive_pairs['bug_hash']) \
                    .union(negative_pairs['fix_hash']).union(negative_pairs['bug_hash'])

iterator = tqdm(os.listdir(pickle_dir)[56:76])

# Loop through all the files in the directory
for file_name in iterator:

    # Check if the file is a pickle
    if file_name.endswith('.pickle'):

        # Get the full path of the pickle file
        pickle_file = os.path.join(pickle_dir, file_name)

        with open(pickle_file, 'rb') as f:
            try:
                data = pickle.load(f)

                for sha in data:

                    # Check if sha is in either fix_hash or bug_hash columns
                    if sha in relevant_shas:
                        COMMIT_LOOKUP[sha] = data[sha]

                # Close the file
                f.close()
                del data

            except Exception as e:
                print(pickle_file, e)

        # Garbage collect
        gc.collect()

print(len(COMMIT_LOOKUP))

# Save COMMIT_LOOKUP as a single regular pickle file
with open(output_file, 'wb') as outfile:
    pickle.dump(COMMIT_LOOKUP, outfile)

# Close the file and garbage collect
outfile.close()
COMMIT_LOOKUP = {}
gc.collect()
