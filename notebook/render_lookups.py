import os
import pickle
from tqdm.auto import tqdm
from dataset import raw_to_padded, _preload, get_labelled, get_labelled_data, POSITIVE_CSV_FILE, NEGATIVE_CSV_FILE
import numpy as np
import random
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

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


    X_train, y_train = get_labelled_data(POSITIVE_CSV_FILE, BAG_SIZE=BAG_SIZE, CONTEXT_SIZE=CONTEXT_SIZE)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
   
    os.makedirs(output_dir, exist_ok=True)
    num_files = (len(X_train) + len(X_test)) // SPLIT + 1
    for i in range(num_files):
        filename = os.path.join(output_dir, f'rendered_labelled_p{i}.pickle')
        start_idx = i * SPLIT
        end_idx = (i + 1) * SPLIT
        X_train_subset = X_train[start_idx:end_idx]
        X_test_subset = X_test[start_idx:end_idx]
        y_train_subset = y_train[start_idx:end_idx]
        y_test_subset = y_test[start_idx:end_idx]
        with open(filename, 'wb') as f_out:
            pickle.dump([X_train_subset, X_test_subset, y_train_subset, y_test_subset], f_out)

    del X_test, y_test, X_train, y_train

    X_train, y_train = get_labelled_data(NEGATIVE_CSV_FILE, BAG_SIZE=BAG_SIZE, CONTEXT_SIZE=CONTEXT_SIZE, sliceE=6000)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    num_files = (len(X_train) + len(X_test)) // SPLIT + 1
    for i in range(num_files):
        filename = os.path.join(output_dir, f'rendered_labelled_n{i}.pickle')
        start_idx = i * SPLIT
        end_idx = (i + 1) * SPLIT
        X_train_subset = X_train[start_idx:end_idx]
        X_test_subset = X_test[start_idx:end_idx]
        y_train_subset = y_train[start_idx:end_idx]
        y_test_subset = y_test[start_idx:end_idx]
        with open(filename, 'wb') as f_out:
            pickle.dump([X_train_subset, X_test_subset, y_train_subset, y_test_subset], f_out)

    del X_test, y_test, X_train, y_train

    X_train, y_train = get_labelled_data(NEGATIVE_CSV_FILE, BAG_SIZE=BAG_SIZE, CONTEXT_SIZE=CONTEXT_SIZE, sliceS=6000)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    num_files = (len(X_train) + len(X_test)) // SPLIT + 1
    for i in range(num_files):
        filename = os.path.join(output_dir, f'rendered_labelled_nn{i}.pickle')
        start_idx = i * SPLIT
        end_idx = (i + 1) * SPLIT
        X_train_subset = X_train[start_idx:end_idx]
        X_test_subset = X_test[start_idx:end_idx]
        y_train_subset = y_train[start_idx:end_idx]
        y_test_subset = y_test[start_idx:end_idx]
        with open(filename, 'wb') as f_out:
            pickle.dump([X_train_subset, X_test_subset, y_train_subset, y_test_subset], f_out)


def oversample_data(X, y):
    # Separate the features
    X_name = np.array([tup[0] for tup in X])
    X_timestamp = np.array([tup[1] for tup in X])
    X_message = np.array([tup[2] for tup in X])
    X_bag1 = np.array([tup[3] for tup in X])
    X_bag2 = np.array([tup[4] for tup in X])

    # Apply Random Oversampling to each feature
    ros = RandomOverSampler()
    indices_resampled, y_resampled = ros.fit_resample(np.arange(len(y)).reshape(-1, 1), y)
    indices_resampled = indices_resampled.flatten()
    X_name = X_name[indices_resampled]
    X_timestamp = X_timestamp[indices_resampled]
    X_message = X_message[indices_resampled]
    X_bag1 = X_bag1[indices_resampled]
    X_bag2 = X_bag2[indices_resampled]

    # Reassemble the tuples
    X_resampled = list(zip(X_name, X_timestamp, X_message, X_bag1, X_bag2))
    return X_resampled, y_resampled


def get_rendered_examples(balance=False, input_dir='../data/commit_lookups/rendered_labelled'):
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    X = []
    y = []

    for filename in os.listdir(input_dir):
        with open(os.path.join(input_dir, filename), 'rb') as f_in:
            data = pickle.load(f_in)
            if(len(data) == 4):
                print("len 4")
                #print((data[0][1][3]))
                X.extend(data[0])
                X.extend(data[1])
                y.extend(data[2])
                y.extend(data[3])
            elif(len(data) == 2):
                print("len 2")
                #print((data[0][1][3]))
                X.extend(data[0])
                y.extend(data[1])
                #X_train.extend(data[0][0])
                #y_train.extend(data[1])
        

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    #print([tup[0] for tup in X_train[:10]])
    #print(X_train[1])
    #print(X_test[1])

            
    
    if(balance):
        #This would be undersampling / downsampling:
        # X_train, y_train = zip(*random.sample([(x, y) for x, y in zip(X_train, y_train) if y == 0], sum(y_train)) + [(x, y) for x, y in zip(X_train, y_train) if y == 1])
        # X_test, y_test = zip(*random.sample([(x, y) for x, y in zip(X_test, y_test) if y == 0], sum(y_test)) + [(x, y) for x, y in zip(X_test, y_test) if y == 1])

        # Apply oversampling to both training and test data
        X_train, y_train = oversample_data(X_train, y_train)
        X_test, y_test = oversample_data(X_test, y_test)


    return X_train, X_test, y_train, y_test


def test():
    X_train, X_test, y_train, y_test = get_rendered_examples(balance=True) #You can also set balance=True (!)
    X_test_1 = [X_test[i] for i in range(len(X_test)) if y_test[i] == 1]
    y_test_1 = [y_test[i] for i in range(len(X_test)) if y_test[i] == 1]
    X_test_0 = [X_test[i] for i in range(len(X_test)) if y_test[i] == 0]
    y_test_0 = [y_test[i] for i in range(len(X_test)) if y_test[i] == 0]


#test()