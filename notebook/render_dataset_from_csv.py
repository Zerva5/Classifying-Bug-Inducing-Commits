import os
import tensorflow as tf
from dataset import raw_to_padded
import pickle
from tqdm.auto import tqdm
from Commit import CommitFactory
Commit = CommitFactory()
import gc
import re
import pandas as pd
import numpy as np
import sys
from process_metadata import processNames, processTimestamps, processCommitMessages

def get_filename(matches_dict, hash):
    row = matches_dict[matches_dict['commit_hash'] == hash][['file_name']]

    return row

    

def make_example(fix_lookup, bug_lookup, BAG_SIZE = 128, CONTEXT_SIZE = 8):
    return [
        np.array(processNames(fix_lookup.author, bug_lookup.author)),
        np.array(processTimestamps(fix_lookup.datetime.timestamp(), bug_lookup.datetime.timestamp())),
        np.array(processCommitMessages(fix_lookup.message, bug_lookup.message)),
        np.array(raw_to_padded(fix_lookup.bag_of_contexts, BAG_SIZE=BAG_SIZE, CONTEXT_SIZE=CONTEXT_SIZE)),
        np.array(raw_to_padded(bug_lookup.bag_of_contexts, BAG_SIZE=BAG_SIZE, CONTEXT_SIZE=CONTEXT_SIZE))
    ]

def createPickleIndexFile(pickle_dir='../data/commit_lookups/unrendered', output_dir='../data/commit_lookups'):
    file_names = os.listdir(pickle_dir)

    ## loop through all files:
    # create a big list of the keys in the pickle files

    keyList = []
    failed_files = []


    for file_name in file_names:
        print("Loading pickle file: {}. Pickle files left: {}".format(file_name, len(file_names) - file_names.index(file_name)))
        with open(os.path.join(pickle_dir, file_name), 'rb') as f:
            try:
                data = pickle.load(f)
                f.close()
                keyDF = pd.DataFrame(data.keys())
                keyDF['file_name'] = file_name
                keyList.append(keyDF)

                del data
                gc.collect()
            except:
                failed_files.append(file_name)

        
    keyDF = pd.concat(keyList)
    # set column names
    keyDF.columns = ['commit_hash', 'file_name']

    # save to csv
    keyDF.to_csv(os.path.join(output_dir, 'hash_to_filename.csv'), index=False)
    print("Failed files: {}".format(failed_files))

### Render a dataset from a CSV file
def render_dataset(csv_file, BAG_SIZE = 128, CONTEXT_SIZE = 8, max_commits = None, max_commit_bag_size = None, verbose = True, 
pickle_dir = '../data/commit_lookups/unrendered', max_pickle_length = 10000, output_dir = '../data/commit_lookups/rendered_labelled', output_name = 'rendered_dataset.pickle'):
    print("Starting to render dataset")

    ## The csv columns we care about are: fix_hash, bug_hash, fix_index, bug_index
    ## The commit data is stored in pickles in the commit_lookups directory and are titles: commit_data_lookup{start_index}-{end_index}.pickle
    ## The commit data is stored in a dictionary with the key being the commit hash and the value being the commit array (I think)

    # go through pickle files and parse their file names to determine what slices we have
    file_names = os.listdir(pickle_dir)

    ## loop through all files:
    # create a big list of the keys in the pickle files

    keyDF = pd.read_csv("../data/hash_to_filename.csv")
    file_names = keyDF['file_name'].unique()

    # load the csv file
    df = pd.read_csv(csv_file)
    # select 20% of the data
    #df = df.sample(frac=0.2)
    df['parsed'] = False

    # precalculate the file names for each entry in the csv file
    df = df.merge(keyDF, how='inner', right_on=['commit_hash'], left_on=['fix_hash'])
    df = df.drop(columns=['commit_hash'])
    df = df.rename(columns={"file_name": "fix_file_name"})

    df = df.merge(keyDF, how='inner', right_on=['commit_hash'], left_on=['bug_hash'])
    df = df.drop(columns=['commit_hash'])
    df = df.rename(columns={"file_name": "bug_file_name"})

    ## get all the unique file names for the fix_index 
    unique_file_names = df['fix_file_name'].unique()
    ## get all the unique file names for the bug_index
    unique_file_names = df['bug_file_name'].unique()
    ## get union
    unique_file_names = list(set(unique_file_names))
    
    print("Unique file names: {}".format(len(unique_file_names)))

    # print occurance of file names in the csv file
    print(df['fix_file_name'].value_counts())

    ## Do we loop through the csv file or the unique file names?
    ## I think we loop through the unique file names and then for each file name we loop through the csv file
    ## But, each csv line might need two pickles, one for the fix and one for the bug
    ## So, we need to loop through the csv file and then for each line we need to load the two pickles
    ## But loading the pickles is really slow, we want to load 2 of them and then process as many pairs as we can
    ## So, we need to load the pickles and then loop through the csv file and process as many pairs as we can

    save_X = []
    save_Y = []

    waiting_for_later = {}

    first_missed_commit = None

    # loop through unique file names
    for file_name in tqdm(unique_file_names):

        with open(os.path.join(pickle_dir, file_name), 'rb') as f:
            if(verbose):
                print("Loading pickle file: " + file_name)

            data = pickle.load(f)
            f.close()

            if(verbose):
                print("Loaded pickle file: " + file_name)

            ## get a subset of the df where the fix file name or the bug file name is the current file name and the pair has not been parsed
            fix_df = df[(df['fix_file_name'] == file_name) | (df['bug_file_name'] == file_name) & (df['parsed'] == False)]
            print("rows with file name {}: {}".format(file_name, fix_df.shape[0]))

            ## get unique file names for the bug_file_name
            unique_bug_file_names = fix_df['bug_file_name'].unique()

            # first we get the pairs that have the same fix_file_name and bug_file_name in fix_df
            easy_pairs = fix_df[(fix_df['bug_file_name'] == file_name) & (fix_df['fix_file_name'] == file_name)]

            # then we get the pairs that are xor of the fix_file_name and bug_file_name in fix_df
            hard_pairs = fix_df[((fix_df['bug_file_name'] == file_name) ^ (fix_df['fix_file_name'] == file_name)) ]

            missing_fix_hashes = 0
            missing_bug_hashes = 0


            # loop through the easy pairs
            if(verbose):
                print("Processing {} easy pairs".format(easy_pairs.shape[0]))
            for index, row in easy_pairs.iterrows():
                print(row['fix_hash'], row['bug_hash'])
                try:
                    data[row['fix_hash']]
                except:
                    missing_fix_hashes += 1
                    continue
                
                try:
                    data[row['bug_hash']]
                except:
                    missing_bug_hashes += 1
                    continue

                example = make_example(data[row['fix_hash']], data[row['bug_hash']], BAG_SIZE=BAG_SIZE, CONTEXT_SIZE=CONTEXT_SIZE)
                df.at[index, 'parsed'] = True
                save_X.append(example)
                save_Y.append(row['Y'])

            if(verbose):
                print("Missing fix hashes: {} and Missing bug hashed: {}. From file {}".format(missing_fix_hashes, missing_bug_hashes, file_name))
                print("Processing {} hard pairs".format(hard_pairs.shape[0]))
    
            print("Adding hard pairs to waiting for later")
            # loop through the hard pairs
            for index, row in hard_pairs.iterrows():
                # we need to add the fix hash and data along with the bug hash to the waiting for later
                if(row['fix_file_name'] == file_name):
                    if(row['bug_file_name'] not in waiting_for_later.keys()):
                        waiting_for_later[row['bug_file_name']] = []

                    waiting_for_later[row['bug_file_name']].append(("bug", data[row['fix_hash']], row['bug_hash'], row['Y']))
                
                elif(row['bug_file_name'] == file_name):
                    if(row['fix_file_name'] not in waiting_for_later.keys()):
                        waiting_for_later[row['fix_file_name']] = []

                    waiting_for_later[row['fix_file_name']].append(("fix", row['fix_hash'], data[row['bug_hash']], row['Y']))



            print("Memory used by save_X: {} and save_Y: {}".format(sys.getsizeof(save_X)/1000000, sys.getsizeof(save_Y)/1000000))
            print("Memory used by waiting_for_later: {}".format(sys.getsizeof(waiting_for_later)/1000000))


            if(file_name in waiting_for_later.keys()):

                # processing waiting for later pairs for the current file name
                finished_waiting = waiting_for_later[file_name]

                
                print("Processsing {} pairs from waiting list".format(len(finished_waiting)))

                for pair in finished_waiting:
                    if pair[0] == "bug":
                        if(verbose):
                            print("Missing bug: {}".format(pair[2]))
                        example = make_example(pair[1], data[pair[2]], BAG_SIZE=BAG_SIZE, CONTEXT_SIZE=CONTEXT_SIZE)
                    elif pair[0] == "fix":
                        if(verbose):
                            print("Missing fix: {}".format(pair[1]))

                        example = make_example(data[pair[1]], pair[2], BAG_SIZE=BAG_SIZE, CONTEXT_SIZE=CONTEXT_SIZE)
                    
                    save_X.append(example)
                    save_Y.append(pair[3])


                # remove the current file name from the waiting for later
                del waiting_for_later[file_name]

            del data
            gc.collect()





    pickle.dump([save_X, save_Y], open(os.path.join(output_dir, output_name), 'wb'))











# get imput and output filename from command line
input_file = sys.argv[1]
output_name = sys.argv[2]
    

#createPickleIndexFile('../data/commit_lookups/unrendered')
render_dataset(input_file, BAG_SIZE=128, CONTEXT_SIZE=8, verbose = True, pickle_dir = '../data/commit_lookups/unrendered', output_name = output_name)

