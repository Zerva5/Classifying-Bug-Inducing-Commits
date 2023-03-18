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
import gzip

def raw_to_padded(bag_of_contexts, CONTEXT_SIZE, BAG_SIZE):

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


with gzip.open('../data/commit_lookups/full_commit_data_lookup.pickle.gz', 'wb') as file:
    pickle.dump(COMMIT_DATA_LOOKUP, file, protocol=pickle.HIGHEST_PROTOCOL)
