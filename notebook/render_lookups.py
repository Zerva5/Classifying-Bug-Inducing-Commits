import os
import pickle
from tqdm.auto import tqdm
from dataset import raw_to_padded, _preload, get_labelled
import numpy as np
import random

from Commit import CommitFactory
Commit = CommitFactory()
import gc

def render_lookups(pickle_dir='../data/commit_lookups/unlabelled', output_dir='../data/commit_lookups/rendered_unlabelled', BAG_SIZE=256, CONTEXT_SIZE=16, verbose=True, max_commit_bag_size=None):
    files = os.listdir(pickle_dir)
    files.sort()

    for file_name in tqdm(files):
        if file_name.endswith('.pickle'):
            input_file_path = os.path.join(pickle_dir, file_name)
            output_file_path = os.path.join(output_dir, file_name)

            if verbose:
                print("Loading file", input_file_path)

            with open(input_file_path, 'rb') as f_in:
                try:
                    data = pickle.load(f_in)
                    processed_data = []
                    for sha in list(data):
                        if max_commit_bag_size is None or len(data[sha].bag_of_contexts) <= max_commit_bag_size:
                            processed_data.append(raw_to_padded(data[sha].bag_of_contexts, BAG_SIZE=BAG_SIZE, CONTEXT_SIZE=CONTEXT_SIZE))
                            del data[sha]

                    if verbose:
                        print("Processed file: {} - Total Commits: {}".format(input_file_path, len(processed_data)))

                    with open(output_file_path, 'wb') as f_out:
                        pickle.dump(processed_data, f_out)

                    del data, processed_data
                    gc.collect()

                except Exception as e:
                    print(input_file_path, e)

SPLIT = 1000

def render_examples(output_dir='../data/commit_lookups/rendered_labelled', BAG_SIZE=256, CONTEXT_SIZE=16, verbose=True):
    _preload()
    X_train, X_test, y_train, y_test = get_labelled(BAG_SIZE=BAG_SIZE, CONTEXT_SIZE=CONTEXT_SIZE)
    os.makedirs(output_dir, exist_ok=True)
    num_files = (len(X_train) + len(X_test)) // SPLIT + 1
    for i in range(num_files):
        filename = os.path.join(output_dir, f'rendered_labelled_{i}.pickle')
        start_idx = i * SPLIT
        end_idx = (i + 1) * SPLIT
        X_train_subset = X_train[start_idx:end_idx]
        X_test_subset = X_test[start_idx:end_idx]
        y_train_subset = y_train[start_idx:end_idx]
        y_test_subset = y_test[start_idx:end_idx]
        with open(filename, 'wb') as f_out:
            pickle.dump([X_train_subset, X_test_subset, y_train_subset, y_test_subset], f_out)

def get_rendered_examples(balance=False, input_dir='../data/commit_lookups/rendered_labelled'):
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for filename in os.listdir(input_dir):
        with open(os.path.join(input_dir, filename), 'rb') as f_in:
            data = pickle.load(f_in)
            X_train.extend(data[0])
            X_test.extend(data[1])
            y_train.extend(data[2])
            y_test.extend(data[3])
    
    if(balance):
        X_train, y_train = zip(*random.sample([(x, y) for x, y in zip(X_train, y_train) if y == 0], sum(y_train)) + [(x, y) for x, y in zip(X_train, y_train) if y == 1])
        X_test, y_test = zip(*random.sample([(x, y) for x, y in zip(X_test, y_test) if y == 0], sum(y_test)) + [(x, y) for x, y in zip(X_test, y_test) if y == 1])

    return X_train, X_test, y_train, y_test
