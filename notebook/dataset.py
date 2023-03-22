##################################
# Note: Three functions to load:
# - Unsupervised examples array
# - Labelled Positives array
# - Labelled Negatives array
##################################

from load_lookups import load_commit_lookup
from process_metadata import processNames, processTimestamps, processCommitMessages
from tqdm.auto import tqdm
import random
import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split
import os
import pickle

from Commit import CommitFactory
Commit = CommitFactory()

#############################
POSITIVE_CSV_FILE = "./apache_positive_pairs.csv"
NEGATIVE_CSV_FILE = "./apache_negative_pairs.csv"
#############################

global COMMIT_LOOKUP
COMMIT_LOOKUP = {}

def _preload(max_commits = None, max_commit_bag_size = None):
    global COMMIT_LOOKUP
    if len(COMMIT_LOOKUP) == 0:
        print("Loading Commit lookup table")
        COMMIT_LOOKUP = load_commit_lookup(max_commits = max_commits, max_commit_bag_size = max_commit_bag_size)

def _unload():
    global COMMIT_DATA_LOOKUP
    del COMMIT_DATA_LOOKUP
    COMMIT_DATA_LOOKUP = {}
    gc.collect()

####################################################################################################################

def raw_to_padded(bag_of_contexts, BAG_SIZE = 256, CONTEXT_SIZE = 16):

    padded_one_hot_paths = []
    for path in bag_of_contexts:
        if len(path) > CONTEXT_SIZE:
            continue
        padded_path = [0] * max(CONTEXT_SIZE - len(path), 0) + path[-CONTEXT_SIZE:]
        padded_one_hot_paths.append(padded_path)

    bag_of_contexts = padded_one_hot_paths

    if(len(bag_of_contexts) == BAG_SIZE):
        return bag_of_contexts

    if(len(bag_of_contexts) > BAG_SIZE):
        return random.sample(bag_of_contexts, BAG_SIZE)

    blank_path = ([0] * CONTEXT_SIZE)

    return ([blank_path] * (BAG_SIZE - len(bag_of_contexts)) + bag_of_contexts)

def row_to_example(row, BAG_SIZE = 256, CONTEXT_SIZE = 16):
    global COMMIT_LOOKUP
    return [
        np.array(processNames(COMMIT_LOOKUP[row['fix_hash']].author, COMMIT_LOOKUP[row['bug_hash']].author)),
        np.array(processTimestamps(COMMIT_LOOKUP[row['fix_hash']].datetime.timestamp(), COMMIT_LOOKUP[row['bug_hash']].datetime.timestamp())),
        np.array(processCommitMessages(COMMIT_LOOKUP[row['fix_hash']].message, COMMIT_LOOKUP[row['bug_hash']].message)),
        np.array(raw_to_padded(COMMIT_LOOKUP[row['fix_hash']].bag_of_contexts, BAG_SIZE=BAG_SIZE, CONTEXT_SIZE=CONTEXT_SIZE)),
        np.array(raw_to_padded(COMMIT_LOOKUP[row['bug_hash']].bag_of_contexts, BAG_SIZE=BAG_SIZE, CONTEXT_SIZE=CONTEXT_SIZE))
    ]

####################################################################################################################


UNLABELLED_PATH="../data/commit_lookups/unlabelled"
def unlabelled_generator(BAG_SIZE=256, CONTEXT_SIZE=16, batch_size=10):
    files = os.listdir(UNLABELLED_PATH)
    files.sort()

    for file_name in files:
        with open(os.path.join(UNLABELLED_PATH, file_name), 'rb') as f:

            while True:
                try:
                    batch = pickle.load(f)
                except EOFError:
                    break

                #for sha in tqdm(batch, total=len(batch), desc="Generating Unsupervised X_train"):
                for sha in batch.keys():
                    X_train = raw_to_padded(batch[sha].bag_of_contexts, BAG_SIZE=BAG_SIZE, CONTEXT_SIZE=CONTEXT_SIZE)
                    #X_train = np.array(X_train, dtype=np.float32)  # Convert the elements in X_train to float32

                    yield (X_train, X_train)


def get_unlabelled(BAG_SIZE = 256, CONTEXT_SIZE = 16):
    raise DeprecationWarning("USE THE GENERATOR!!!")
    raise Exception("PLS DON'T USE THIS")
    global COMMIT_LOOKUP
    _preload()

    X_train = [raw_to_padded(COMMIT_LOOKUP[sha].bag_of_contexts, BAG_SIZE=BAG_SIZE, CONTEXT_SIZE=CONTEXT_SIZE) for sha in tqdm(COMMIT_LOOKUP, total=len(COMMIT_LOOKUP), desc="Generating Unsupervised X_train")]

    return X_train

# def get_positive_labelled(BAG_SIZE = 256, CONTEXT_SIZE = 16):
#     global COMMIT_LOOKUP
#     _preload()

#     RAW_EXAMPLES = pd.read_csv(POSITIVE_CSV_FILE)
    
#     X_train = [
#         row_to_example(row, BAG_SIZE=BAG_SIZE, CONTEXT_SIZE=CONTEXT_SIZE)
#         for i, row in tqdm(RAW_EXAMPLES.iterrows(), total=len(RAW_EXAMPLES), desc="Generating Positive X_train") if row['fix_hash'] in COMMIT_LOOKUP and row['bug_hash'] in COMMIT_LOOKUP
#     ]

#     y_train = [row["Y"] for i, row in tqdm(RAW_EXAMPLES.iterrows(), total=len(RAW_EXAMPLES), desc="Generating Positive y_train") if row['fix_hash'] in COMMIT_LOOKUP and row['bug_hash'] in COMMIT_LOOKUP]

#     return X_train, y_train

# def get_negative_labelled(BAG_SIZE = 256, CONTEXT_SIZE = 16):
#     global COMMIT_LOOKUP
#     _preload()

#     RAW_EXAMPLES = pd.read_csv(NEGATIVE_CSV_FILE)
    
#     X_train = [
#         row_to_example(row, BAG_SIZE=BAG_SIZE, CONTEXT_SIZE=CONTEXT_SIZE)
#         for i, row in tqdm(RAW_EXAMPLES.iterrows(), total=len(RAW_EXAMPLES), desc="Generating Negative X_train") if row['fix_hash'] in COMMIT_LOOKUP and row['bug_hash'] in COMMIT_LOOKUP
#     ]

#     y_train = [row["Y"] for i, row in tqdm(RAW_EXAMPLES.iterrows(), total=len(RAW_EXAMPLES), desc="Generating Negative y_train") if row['fix_hash'] in COMMIT_LOOKUP and row['bug_hash'] in COMMIT_LOOKUP]

#     return X_train, y_train

import multiprocessing as mp


def row_to_example_helper(args):
    return row_to_example(args[0], BAG_SIZE=args[1], CONTEXT_SIZE=args[2]), args[0]["Y"]

def get_labelled_data(file, BAG_SIZE=256, CONTEXT_SIZE=16):
    global COMMIT_LOOKUP
    _preload()

    RAW_EXAMPLES = pd.read_csv(file)

    # Prepare arguments for row_to_example_helper
    args = [(row, BAG_SIZE, CONTEXT_SIZE) for _, row in RAW_EXAMPLES.iterrows() if row['fix_hash'] in COMMIT_LOOKUP and row['bug_hash'] in COMMIT_LOOKUP]

    # Use multiprocessing to parallelize the list comprehension
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(row_to_example_helper, args), total=len(args), desc=f"Generating {file.split('/')[-1].split('.')[0]} X_train and y_train"))

    X_train, y_train = zip(*results)
    return list(X_train), list(y_train)

def get_positive_labelled(BAG_SIZE=256, CONTEXT_SIZE=16):
    return get_labelled_data(POSITIVE_CSV_FILE, BAG_SIZE=BAG_SIZE, CONTEXT_SIZE=CONTEXT_SIZE)

def get_negative_labelled(BAG_SIZE=256, CONTEXT_SIZE=16):
    return get_labelled_data(NEGATIVE_CSV_FILE, BAG_SIZE=BAG_SIZE, CONTEXT_SIZE=CONTEXT_SIZE)

def get_labelled(BAG_SIZE=256, CONTEXT_SIZE=16):
    global COMMIT_LOOKUP
    _preload()

    # Get positive and negative labelled data
    X_train_positive, y_train_positive = get_labelled_data(POSITIVE_CSV_FILE, BAG_SIZE=BAG_SIZE, CONTEXT_SIZE=CONTEXT_SIZE)
    X_train_negative, y_train_negative = get_labelled_data(NEGATIVE_CSV_FILE, BAG_SIZE=BAG_SIZE, CONTEXT_SIZE=CONTEXT_SIZE)

    X_train = X_train_positive + X_train_negative
    y_train = y_train_positive + y_train_negative

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

