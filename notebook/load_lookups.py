import os
import pickle
from tqdm.auto import tqdm
from Commit import CommitFactory
Commit = CommitFactory()

#TODO: Make this process more efficient @lucas
def load_commit_lookup(pickle_dir = '../data/commit_lookups', verbose = True, max_commits = None, max_commit_bag_size = None):

    COMMIT_DATA_LOOKUP = {}

    iterator = tqdm(os.listdir(pickle_dir))
    
    # loop through all the files in the directory
    for file_name in iterator:

        # check if the file is a pickle
        if file_name.endswith('.pickle'):

            # get the full path of the pickle file
            pickle_file = os.path.join(pickle_dir, file_name)

            if(verbose):
                print("Loading file", pickle_file)

            with open(pickle_file, 'rb') as f:
                try:
                    data = pickle.load(f)

                    for sha in data:
                        if max_commits != None and len(COMMIT_DATA_LOOKUP) > max_commits:
                            iterator.close()
                            return COMMIT_DATA_LOOKUP

                        if(max_commit_bag_size == None or len(data[sha].bag_of_contexts) <= max_commit_bag_size):
                            COMMIT_DATA_LOOKUP[sha] = data[sha]

                    if(verbose):
                        print("Appending pickle of length:", len(data.keys()), ", new dict length:", len(COMMIT_DATA_LOOKUP.keys()))
                except Exception as e:
                    print(pickle_file, e)

    return COMMIT_DATA_LOOKUP