import os
import pickle
from tqdm.auto import tqdm
from Commit import CommitFactory
Commit = CommitFactory()
import gc
import re
import pandas as pd

def get_filename(matches_dict, hash):
    row = matches_dict[matches_dict['commit_hash'] == hash][['file_name']]

    return row

    

def make_example(fix_lookup, bug_lookup, BAG_SIZE = 256, CONTEXT_SIZE = 16):
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
def render_dataset(csv_file, BAG_SIZE = 256, CONTEXT_SIZE = 16, max_commits = None, max_commit_bag_size = None, verbose = True, pickle_dir = '../data/commit_lookups/unrendered', max_pickle_length = 10000):
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


    ## Do we loop through the csv file or the unique file names?
    ## I think we loop through the unique file names and then for each file name we loop through the csv file
    ## But, each csv line might need two pickles, one for the fix and one for the bug
    ## So, we need to loop through the csv file and then for each line we need to load the two pickles
    ## But loading the pickles is really slow, we want to load 2 of them and then process as many pairs as we can
    ## So, we need to load the pickles and then loop through the csv file and process as many pairs as we can

    save_pairs = []

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

            ## get a subset of the df where the fix_file_name is the current file_name
            fix_df = df[df['fix_file_name'] == file_name]

            ## get unique file names for the bug_file_name
            unique_bug_file_names = fix_df['bug_file_name'].unique()

            # first we get the pairs that have the same fix_file_name and bug_file_name in fix_df
            easy_pairs = fix_df[(fix_df['bug_file_name'] == file_name)]

            # then we get the pairs that have different fix_file_name and bug_file_name in fix_df
            hard_pairs = fix_df[fix_df['bug_file_name'] != file_name]

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
                
                try:
                    data[row['bug_hash']]
                except:
                    missing_bug_hashes += 1
                #example = make_example(data[row['fix_hash']], data[row['bug_hash']], BAG_SIZE=BAG_SIZE, CONTEXT_SIZE=CONTEXT_SIZE)
                #df.at[index, 'parsed'] = True
                #save_pairs.append(example)

                ## if we have hit the pairs limit, save the pairs and reset the save_pairs list
                if(len(save_pairs) >= max_pickle_length):
                    ##save(save_pairs, pickle_dir)
                    print("SAVING PAIRS (not actually saving)")
                    del save_pairs
                    save_pairs = []
            print("Missing fix hashes: {} and Missing bug hashed: {}. From file {}".format(missing_fix_hashes, missing_bug_hashes, file_name))












    

#createPickleIndexFile('../data/commit_lookups/unrendered')
# render_dataset("../data/pairs_output/apache_close_strict_negative_pairs.csv", max_commits = 1000, max_commit_bag_size = 256, verbose = True, pickle_dir = '../data/commit_lookups/unrendered')


#TODO: Make this process more efficient @lucas
def load_commit_lookup(pickle_dir = '../data/commit_lookups/labelled', verbose = True, max_commits = None, max_commit_bag_size = None):

    COMMIT_DATA_LOOKUP = {}

    files = os.listdir(pickle_dir)
    files.sort()


    if(max_commits == None):
        total = len(files)
        desc = "Loading commit lookup files"
    else:
        total = max_commits
        desc = "Loading commit lookups"
    with tqdm(total=total, desc=desc) as pbar:

        # loop through all the files in the directory
        for file_name in files:

            # check if the file is a pickle
            if file_name.endswith('.pickle'):

                # get the full path of the pickle file
                pickle_file = os.path.join(pickle_dir, file_name)

                if(verbose):
                    print("Loading file", pickle_file)

                with open(pickle_file, 'rb') as f:
                    try:
                        data = pickle.load(f)
                        data_size = len(data)
                        for sha in list(data):
                            if max_commits != None and len(COMMIT_DATA_LOOKUP) >= max_commits:
                                return COMMIT_DATA_LOOKUP

                            if(max_commit_bag_size == None or len(data[sha].bag_of_contexts) <= max_commit_bag_size):
                                COMMIT_DATA_LOOKUP[sha] = data[sha]
                                del data[sha]
                                if(max_commits != None):
                                    pbar.update(1)

                        if(verbose):
                            print("Appending pickle of length:", data_size, ", new dict length:", len(COMMIT_DATA_LOOKUP.keys()))

                        f.close()
                        del data
                        gc.collect()
                        
                    except Exception as e:
                        print(pickle_file, e)
                
                if(max_commits == None):
                    pbar.update(1)

    return COMMIT_DATA_LOOKUP
