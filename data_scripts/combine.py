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

# specify the directory containing the pickles
pickle_dir = '../data/commit_lookups'

COMMIT_DATA_LOOKUP = {}

# loop through all the files in the directory
for file_name in os.listdir(pickle_dir):
    # check if the file is a pickle
    if file_name.endswith('.pickle'):
        # get the full path of the pickle file
        pickle_file = os.path.join(pickle_dir, file_name)

        with(open(pickle_file, 'rb') as f):
            data = pickle.load(f)        
            COMMIT_DATA_LOOKUP.update(data)
            print("Appending pickle of length:", len(data.keys()), ", new dict length:", len(COMMIT_DATA_LOOKUP.keys()))
            
