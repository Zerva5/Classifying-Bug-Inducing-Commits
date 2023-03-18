import sys
import ast
import git
import pickle
import numpy as np
import random
from tqdm.auto import tqdm, trange
import os
from datetime import datetime
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, ALL_COMPLETED
import networkx as nx
import tree_sitter
from tree_sitter import Language
import json
import requests
import pandas as pd
from collections import defaultdict
from github import Github
import base64
import cProfile
import pstats
import time
import re

#Toggle to determine if tokens should be one-hot encoded BY OUR PRE-PROCESSING or not
ONE_HOT = False

#Toggle to drop paths above the context size limit. Likely should always be set to true.
IGNORE_DEEP_PATHS = True

#Generate full tuples of (terminal_token_a, (path...), terminal_token_b) instead of just paths.
#commit2seq describes using this strategy as being quite a bit more effective (like 30%), but currently I
#haven't figured out how to implement it.
USE_FULL_TUPLES = False    #NOT working yet

#Set this to true in order to speed up path generation. However makes it so pruning to the BAG_SIZE happens
#immediately and can't be adjusted without a re-run. If this is false the pruning and padding must be done before training
IMMEDIATE_PAD_AND_PRUNE = False


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

# specify the directory containing the pickles
pickle_dir = '../data/commit_lookups'

COMMIT_DATA_LOOKUP = {}

# loop through all the files in the directory
for file_name in os.listdir(pickle_dir):
    # check if the file is a pickle
    if file_name.endswith('.pickle'):
        # get the full path of the pickle file
        pickle_file = os.path.join(pickle_dir, file_name)

        with open(pickle_file, 'rb') as f:
            try:
                data = pickle.load(f)        
                COMMIT_DATA_LOOKUP.update(data)
                print("Appending pickle of length:", len(data.keys()), ", new dict length:", len(COMMIT_DATA_LOOKUP.keys()))
            except Exception as e:
                print(pickle_file, e)


with open('../data/commit_lookups/full_commit_data_lookup.pickle', 'wb') as file:
    pickle.dump(COMMIT_DATA_LOOKUP, file, protocol=pickle.HIGHEST_PROTOCOL)