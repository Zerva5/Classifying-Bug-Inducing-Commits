import os
import pickle
from Commit import CommitFactory
Commit = CommitFactory()

#TODO: Make this process more efficient @lucas
def load_commit_lookup(pickle_dir = '../data/commit_lookups', verbose = True):

    COMMIT_DATA_LOOKUP = {}
    
    # loop through all the files in the directory
    for file_name in os.listdir(pickle_dir):
        # check if the file is a pickle
        if file_name.endswith('.pickle'):

            # get the full path of the pickle file
            pickle_file = os.path.join(pickle_dir, file_name)

            if(verbose):
                print("Loading file", pickle_file)

            with open(pickle_file, 'rb') as f:
                try:
                    data = pickle.load(f)
                    COMMIT_DATA_LOOKUP.update(data)
                    if(verbose):
                        print("Appending pickle of length:", len(data.keys()), ", new dict length:", len(COMMIT_DATA_LOOKUP.keys()))
                except Exception as e:
                    print(pickle_file, e)

    return COMMIT_DATA_LOOKUP