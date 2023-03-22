import os
import pickle
from tqdm.auto import tqdm
from Commit import CommitFactory
Commit = CommitFactory()
import gc

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
