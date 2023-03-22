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
    global COMMIT_LOOKUP
    del COMMIT_LOOKUP
    COMMIT_LOOKUP = {}
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
        k = len(bag_of_contexts) // BAG_SIZE
        indices = np.arange(0, len(bag_of_contexts), k)[:BAG_SIZE]
        return bag_of_contexts[indices]

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

def get_unlabelled(BAG_SIZE = 256, CONTEXT_SIZE = 16):
    global COMMIT_LOOKUP
    _preload()

    X_train = [raw_to_padded(COMMIT_LOOKUP[sha].bag_of_contexts, BAG_SIZE=BAG_SIZE, CONTEXT_SIZE=CONTEXT_SIZE) for sha in tqdm(COMMIT_LOOKUP, total=len(COMMIT_LOOKUP), desc="Generating Unsupervised X_train")]

    return X_train

def get_positive_labelled(BAG_SIZE = 256, CONTEXT_SIZE = 16):
    global COMMIT_LOOKUP
    _preload()

    RAW_EXAMPLES = pd.read_csv(POSITIVE_CSV_FILE)
    
    X_train = [
        row_to_example(row, BAG_SIZE=BAG_SIZE, CONTEXT_SIZE=CONTEXT_SIZE)
        for i, row in tqdm(RAW_EXAMPLES.iterrows(), total=len(RAW_EXAMPLES), desc="Generating Positive X_train") if row['fix_hash'] in COMMIT_LOOKUP and row['bug_hash'] in COMMIT_LOOKUP
    ]

    y_train = [row["Y"] for i, row in tqdm(RAW_EXAMPLES.iterrows(), total=len(RAW_EXAMPLES), desc="Generating Positive y_train") if row['fix_hash'] in COMMIT_LOOKUP and row['bug_hash'] in COMMIT_LOOKUP]

    return X_train, y_train

def get_negative_labelled(BAG_SIZE = 256, CONTEXT_SIZE = 16):
    global COMMIT_LOOKUP
    _preload()

    RAW_EXAMPLES = pd.read_csv(NEGATIVE_CSV_FILE)
    
    X_train = [
        row_to_example(row, BAG_SIZE=BAG_SIZE, CONTEXT_SIZE=CONTEXT_SIZE)
        for i, row in tqdm(RAW_EXAMPLES.iterrows(), total=len(RAW_EXAMPLES), desc="Generating Negative X_train") if row['fix_hash'] in COMMIT_LOOKUP and row['bug_hash'] in COMMIT_LOOKUP
    ]

    y_train = [row["Y"] for i, row in tqdm(RAW_EXAMPLES.iterrows(), total=len(RAW_EXAMPLES), desc="Generating Negative y_train") if row['fix_hash'] in COMMIT_LOOKUP and row['bug_hash'] in COMMIT_LOOKUP]

    return X_train, y_train

def get_labelled(BAG_SIZE = 256, CONTEXT_SIZE = 16):
    global COMMIT_LOOKUP
    _preload()
    
    X_train_positive, y_train_positive = get_positive_labelled(BAG_SIZE = BAG_SIZE, CONTEXT_SIZE = CONTEXT_SIZE)
    X_train_negative, y_train_negative = get_negative_labelled(BAG_SIZE = BAG_SIZE, CONTEXT_SIZE = CONTEXT_SIZE)

    X_train = X_train_positive + X_train_negative
    y_train = y_train_positive + y_train_negative

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test