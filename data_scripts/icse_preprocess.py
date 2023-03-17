def load_token(filename):
    with open(filename, 'r') as f:
        token = f.read().strip()
    return token

GH_ACCESS_TOKEN = load_token("../gh_access_token.txt")

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
import base64
import requests
API_BASE_URL = "https://api.github.com"

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

#Load language files
#The ast.so file must be re-generated to add support for more languages.
#To do this, see: https://github.com/tree-sitter/py-tree-sitter

JAVA_LANGUAGE = Language('../ast-bindings/build/ast.so', 'java')

def download_json_data(lang):
    url = f"https://raw.githubusercontent.com/tree-sitter/tree-sitter-{lang}/master/src/node-types.json"
    data = json.loads(requests.get(url).text)
    types = [node_type["type"] for node_type in data]
    for node_type in data:
        if "subtypes" in node_type:
            subtypes = [subtype["type"] for subtype in node_type["subtypes"]]
            types.extend(subtypes)
    types = list(set(types))
    return types

JAVA_NODE_TYPES = download_json_data("java")
ALL_NODE_TYPES = JAVA_NODE_TYPES
FILE_FILTERS = (".java")
MAX_NODE_LOOKUP_NUM = len(ALL_NODE_TYPES)
ALL_NODE_INDEXES = range(MAX_NODE_LOOKUP_NUM)

RAW_DATA = pd.read_csv("../data/pairs_output/icse_java_pairs.csv")
# RAW_DATA = pd.read_csv("../data/all_apache_commits.csv")

# RAW_DATA = RAW_DATA.loc[RAW_DATA['diff_line_count'] <= 50]
# RAW_DATA = RAW_DATA.loc[RAW_DATA['files_changed'] <= 8]


REPO_LOOKUP = defaultdict(list)

# repos = set(list(RAW_DATA["repo"]))

# #Create a "clones" directory in order to clone local repos
# if not os.path.exists("../clones"):
#     os.makedirs("../clones")

# # Clone each "repo" into the "clones" folder
# for repo in tqdm(repos):
#     if not os.path.exists(f"../clones/{repo}"):
#         os.system(f"git clone https://github.com/{repo.replace('-', '/')}.git ../clones/{repo}")

#     #TODO: Try to implement this. Currently it breaks concurrant pre-processing
#     # REPO_LOOKUP[f"clones/{repo}"] = git.Repo(f"clones/{repo}")

class Commit:
    
    ###################################### SETUP ###################################### 
    
    def __init__(self, sha, repo_path):
        self.sha = sha
        self.repo_path = repo_path
        self.parent = None
        self.author = None
        self.datetime = None
        self.message = None
        self.bag_of_contexts = None

    def _populate_commit_info(self):
        print("bad4")
        if(self.repo_path in REPO_LOOKUP):
            repo = REPO_LOOKUP[self.repo_path]
        else:
            repo = git.Repo(self.repo_path)
        commit = repo.commit(self.sha)
        self.parent = commit.parents[0].hexsha if commit.parents else None
        self.author = commit.author.name if commit.author else None
        self.datetime = datetime.fromtimestamp(commit.committed_date)
        self.message = commit.message if commit.message else None
    
    def _GH_populate_commit_info(self):
        g = Github(GH_ACCESS_TOKEN)
        repo = g.get_repo(self.repo_path)
        self.repo = repo
        commit = repo.get_commit(sha=self.sha)
        self.parent = commit.parents[0].sha if commit.parents else None
        self.author = commit.author.name if commit.author else None
        self.datetime = commit.commit.author.date
        self.message = commit.commit.message if commit.commit.message else None
        
    def _generate_bags_of_contexts(self):
        if IMMEDIATE_PAD_AND_PRUNE:
            self.bag_of_contexts = self.to_padded_bag_of_contexts()
        else:
            self.bag_of_contexts = self.to_raw_bag_of_contexts()

    ###################################### GIT and GitHub ###################################### 
        
    ################# GIT #################
    
    def get_files_changed(self):
        print("bad1")
        try:
            if(self.repo_path in REPO_LOOKUP):
                repo = REPO_LOOKUP[self.repo_path]
            else:
                repo = git.Repo(self.repo_path)
            commit = repo.commit(self.sha)
            return [diff.a_path for diff in commit.diff(commit.parents[0]) if diff.a_path.endswith(FILE_FILTERS)]
        except:
            return []

    def get_source_at_commit(self, file_name):
        print("bad2")
        try:
            if(self.repo_path in REPO_LOOKUP):
                repo = REPO_LOOKUP[self.repo_path]
            else:
                repo = git.Repo(self.repo_path)
            commit = repo.commit(self.sha)
            return commit.tree[file_name].data_stream.read().decode('utf-8')
        except:
            return ''

    def get_source_at_parent(self, file_name):
        print("bad3")
        try:
            if(self.repo_path in REPO_LOOKUP):
                repo = REPO_LOOKUP[self.repo_path]
            else:
                repo = git.Repo(self.repo_path)
            commit = repo.commit(self.sha)
            return commit.parents[0].tree[file_name].data_stream.read().decode('utf-8')
        except:
            return ''
        
    def extract_method_asts(self, java_ast):
        method_asts = []

        def traverse(node):
            if node.type == "method_declaration":
                method_asts.append(node)
            else:
                for child in node.children:
                    traverse(child)

        traverse(java_ast.root_node)
        return method_asts

    def pair_method_asts(self, pre_commit_ast, post_commit_ast):
        pre_commit_methods = self.extract_method_asts(pre_commit_ast)
        post_commit_methods = self.extract_method_asts(post_commit_ast)

        pre_commit_method_names = [self.get_method_name(ast) for ast in pre_commit_methods]
        post_commit_method_names = [self.get_method_name(ast) for ast in post_commit_methods]

        paired_methods = []

        for pre_commit_method, pre_commit_name in zip(pre_commit_methods, pre_commit_method_names):
            if pre_commit_name in post_commit_method_names:
                index = post_commit_method_names.index(pre_commit_name)
                paired_methods.append((pre_commit_method, post_commit_methods[index]))
                post_commit_methods.pop(index)
                post_commit_method_names.pop(index)
            else:
                paired_methods.append((pre_commit_method, None))

        for post_commit_method in post_commit_methods:
            paired_methods.append((None, post_commit_method))

        return paired_methods

    def get_method_name(self, method_ast):
        for child in method_ast.children:
            if child.type == "identifier":
                return child.text
        return None


    
    ################# GitHub #################
    #### Still todo, in testing ####
    
    def gh_get_files_changed(self):
        try:
            commit = self.repo.get_commit(sha=self.sha)
            return [f.filename for f in commit.files if f.filename.endswith(FILE_FILTERS)]
        except Exception as e:
            print(e)
            return []

    def gh_get_source_at_commit(self, file_name):
        try:
            headers = {'Authorization': f'token {GH_ACCESS_TOKEN}'}
            url = f"{API_BASE_URL}/repos/{self.repo_path}/contents/{file_name}?ref={self.sha}"
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            contents = response.json()
            return base64.b64decode(contents['content']).decode('utf-8')
        except Exception as e:
            print("gh_get_source_at_commit:", e)
            return ''
        
    def gh_get_source_at_parent(self, file_name):
        try:
            headers = {'Authorization': f'token {GH_ACCESS_TOKEN}'}
            url = f"{API_BASE_URL}/repos/{self.repo_path}/contents/{file_name}?ref={self.parent}"
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            contents = response.json()
            return base64.b64decode(contents['content']).decode('utf-8')
        except Exception as e:
            print("gh_get_source_at_parent:", e)
            return ''

    ###################################### AST Processing ###################################### 


    def source_to_ast(self, source, file_name):
        try:
            parser = tree_sitter.Parser()
            if file_name.endswith('.c'):
                parser.set_language(C_LANGUAGE)
            elif file_name.endswith('.cpp'):
                parser.set_language(CPP_LANGUAGE)
            elif file_name.endswith('.java'):
                parser.set_language(JAVA_LANGUAGE)
            elif file_name.endswith('.js'):
                parser.set_language(JS_LANGUAGE)
            elif file_name.endswith('.py'):
                parser.set_language(PY_LANGUAGE)
            else:
                print("UNKNOWN LANGUAGE")
                return None
            return parser.parse(bytes(source, 'utf8'))
        except:
            return None

    def ast_to_graph(self, ast, start_node=None):
        graph = nx.DiGraph()
        node_id = 0
        leaf_nodes = []
        
        if ast is None:
            return graph, leaf_nodes

        def add_node(node):
            nonlocal node_id
            node_name = node.type
            node_text = node.text
            node_id += 1
            graph.add_node(node_id, name=node_name, text=node_text)
            return node_id

        def traverse(node, parent_id=None):
            current_id = add_node(node)
            if parent_id is not None:
                graph.add_edge(parent_id, current_id)

            if not node.children:
                leaf_nodes.append(current_id)
            else:
                for child in node.children:
                    traverse(child, current_id)

        if start_node is None:
            start_node = ast

        traverse(start_node)

        graph.graph['root'] = (start_node.type, start_node.text)  # Store the root node as an attribute

        return graph, leaf_nodes

    
    def prune_identical_subtrees(self, tree1, tree2, leaf_nodes1, leaf_nodes2):
        tree1_copy = tree1.copy()
        tree2_copy = tree2.copy()

        def are_nodes_identical(node1, node2):
            return node1['name'] == node2['name'] and node1['text'] == node2['text']

        def traverse(node1, node2, reduced_leaf_nodes1, reduced_leaf_nodes2):
            children1 = list(tree1_copy.successors(node1))
            children2 = list(tree2_copy.successors(node2))

            if not children1 and not children2:
                if not are_nodes_identical(tree1_copy.nodes[node1], tree2_copy.nodes[node2]):
                    reduced_leaf_nodes1.append(node1)
                    reduced_leaf_nodes2.append(node2)
                return

            identical_children = []

            for child1, child2 in zip(children1, children2):
                traverse(child1, child2, reduced_leaf_nodes1, reduced_leaf_nodes2)
                if are_nodes_identical(tree1_copy.nodes[child1], tree2_copy.nodes[child2]):
                    identical_children.append((child1, child2))

            for child1, child2 in identical_children:
                tree1_copy.remove_node(child1)
                tree2_copy.remove_node(child2)

        root1 = list(tree1_copy.in_degree())[0][0]
        root2 = list(tree2_copy.in_degree())[0][0]
        reduced_leaf_nodes1 = []
        reduced_leaf_nodes2 = []

        traverse(root1, root2, reduced_leaf_nodes1, reduced_leaf_nodes2)

        return tree1_copy, tree2_copy, reduced_leaf_nodes1, reduced_leaf_nodes2

        
    ###################################### Bag of Contexts Processing ###################################### 
        
    def get_paths(self, graph, leaf_nodes, unique_leaf_nodes):
        
        graph = graph.to_undirected()

        all_paths = []
        
        if len(leaf_nodes) == len(unique_leaf_nodes):
            path_lookup = dict(nx.all_pairs_shortest_path(graph))
        
        for terminalA in leaf_nodes:
            for terminalB in unique_leaf_nodes:
                if terminalA == terminalB:
                    continue

                try:
                    if len(leaf_nodes) == len(unique_leaf_nodes):
                        path = path_lookup[terminalA][terminalB]
                    else:
                        path = nx.shortest_path(graph, source=terminalA, target=terminalB) 

                    if(len(path) == 1):
                        continue
                    node_types = [graph.nodes[nodeID]['name'] for nodeID in path]
                    if(USE_FULL_TUPLES):
                        all_paths.append((graph.nodes[pair[0]]['text'], tuple(node_types), graph.nodes[pair[1]]['text']))
                    else:
                        all_paths.append(tuple(node_types))
                except Exception as e:
                    pass
                    
        return set(all_paths)

    def ast_to_bag_of_contexts(self, ast_trees):
        paths = set()
        for tree in ast_trees:
            paths |= self.get_paths(tree)
        return paths

    ###################################### Padding and Encoding ###################################### 
    
    def map_bag_of_contexts_to_id(self, bag_of_contexts):
        mapped_paths = []
        for path in bag_of_contexts:
            mapped_path = []
            for node in path:
                index = ALL_NODE_TYPES.index(node)
                mapped_path.append(index + 1)
            mapped_paths.append(mapped_path)
        return mapped_paths

    def one_hot_encode(self, bag_of_contexts):
        one_hot_paths = []

        # Iterate over each row in the array
        for row in bag_of_contexts:
            # Create an empty list to hold the one-hot encodings for this row
            row_one_hot = []

            # Iterate over each element in the row
            for num in row:
                # Create an empty list to hold the one-hot encoding for this number
                num_one_hot = [0] * (MAX_NODE_LOOKUP_NUM + 1)

                # Set the corresponding element to 1
                num_one_hot[int(num)] = 1

                # Add the one-hot encoding for this number to the row's list
                row_one_hot.append(num_one_hot)

            # Add the row's list of one-hot encodings to the main list
            one_hot_paths.append(row_one_hot)

        return one_hot_paths

    def pad_each_context(self, bag_of_contexts):
        padded_one_hot_paths = []
        for path in bag_of_contexts:
            if IGNORE_DEEP_PATHS and len(path) > CONTEXT_SIZE:
                continue
            if ONE_HOT:
                padded_path = [[0] * (MAX_NODE_LOOKUP_NUM + 1)] * max(CONTEXT_SIZE - len(path), 0) + path[-CONTEXT_SIZE:]
            else:
                padded_path = [0] * max(CONTEXT_SIZE - len(path), 0) + path[-CONTEXT_SIZE:]
            padded_one_hot_paths.append(padded_path)
        return padded_one_hot_paths

    ###################################### Utility ###################################### 
    
    def to_raw_bag_of_contexts(self):
        # files_changed = self.get_files_changed()
        files_changed = self.gh_get_files_changed()

        if(len(files_changed) > 32):
            files_changed = []

        # sources_at_commit = [self.get_source_at_commit(filename) for filename in files_changed]
        # sources_at_parent = [self.get_source_at_parent(filename) for filename in files_changed]

        sources_at_commit = [self.gh_get_source_at_commit(filename) for filename in files_changed]
        sources_at_parent = [self.gh_get_source_at_parent(filename) for filename in files_changed]

        asts_commit = [self.source_to_ast(source, files_changed[i]) for i, source in enumerate(sources_at_commit)]
        asts_parent = [self.source_to_ast(source, files_changed[i]) for i, source in enumerate(sources_at_parent)]

        method_asts_commit = []
        method_asts_parent = []

        for pre_commit_ast, post_commit_ast in zip(asts_parent, asts_commit):
            file_paired_methods = self.pair_method_asts(pre_commit_ast, post_commit_ast)

            for pre_commit_method, post_commit_method in file_paired_methods:
                method_asts_parent.append(pre_commit_method)
                method_asts_commit.append(post_commit_method)

        filtered_method_asts_parent = []
        filtered_method_asts_commit = []

        for parent_ast, commit_ast in zip(method_asts_parent, method_asts_commit):
            if parent_ast is None or commit_ast is None or parent_ast.sexp() != commit_ast.sexp():
                filtered_method_asts_parent.append(parent_ast)
                filtered_method_asts_commit.append(commit_ast)

        graphs_parent = []
        leaf_nodes_parent = []
        graphs_commit = []
        leaf_nodes_commit = []

        # For method ASTs pre-commit
        for method_ast in filtered_method_asts_parent:
            graph, leaf_nodes = self.ast_to_graph(method_ast)
            graphs_parent.append(graph)
            leaf_nodes_parent.append(leaf_nodes)

        # For method ASTs post-commit
        for method_ast in filtered_method_asts_commit:
            graph, leaf_nodes = self.ast_to_graph(method_ast)
            graphs_commit.append(graph)
            leaf_nodes_commit.append(leaf_nodes)

        reduced_asts = []
        reduced_leaf_nodes = []
        original_asts = []
        original_leaf_nodes = []
        for ast_commit, ast_parent, leaf_nodes_commit_i, leaf_nodes_parent_i in zip(graphs_commit, graphs_parent, leaf_nodes_commit, leaf_nodes_parent):
            if ast_commit is None and ast_parent is None:
                continue
            if len(leaf_nodes_commit_i) == 0 and len(leaf_nodes_parent_i) == 0:
                continue
            
            if ast_commit is None or len(leaf_nodes_commit_i) == 0:
                reduced_asts.append(ast_parent)
                reduced_leaf_nodes.append(leaf_nodes_parent_i)

                original_asts.append(ast_parent)
                original_leaf_nodes.append(leaf_nodes_parent_i)
                continue
                
            if ast_parent is None or len(leaf_nodes_parent_i) == 0:
                reduced_asts.append(ast_commit)
                reduced_leaf_nodes.append(leaf_nodes_commit_i)

                original_asts.append(ast_commit)
                original_leaf_nodes.append(leaf_nodes_commit_i)
                continue
            
            reduced_ast_commit, reduced_ast_parent, reduced_leaf_nodes_commit_i, reduced_leaf_nodes_parent_i = self.prune_identical_subtrees(ast_commit, ast_parent, leaf_nodes_commit_i, leaf_nodes_parent_i)
            reduced_asts.append(reduced_ast_commit)
            reduced_asts.append(reduced_ast_parent)
            reduced_leaf_nodes.append(reduced_leaf_nodes_commit_i)
            reduced_leaf_nodes.append(reduced_leaf_nodes_parent_i)
            
            original_asts.append(ast_commit)
            original_asts.append(ast_parent)
            original_leaf_nodes.append(leaf_nodes_commit_i)
            original_leaf_nodes.append(leaf_nodes_parent_i)

            
        contexts = set()
        for i, tree in enumerate(original_asts):
            contexts |= self.get_paths(tree, original_leaf_nodes[i], reduced_leaf_nodes[i])
        # for i, tree in enumerate(reduced_asts):
        #     contexts |= self.get_paths(tree, reduced_leaf_nodes[i], reduced_leaf_nodes[i])

        contexts = self.map_bag_of_contexts_to_id(contexts)

        if(ONE_HOT):
            contexts = self.one_hot_encode(contexts)

        return contexts

    
    def raw_to_padded(self, bag_of_contexts):
        
        bag_of_contexts = self.pad_each_context(bag_of_contexts)

        
        if(len(bag_of_contexts) == BAG_SIZE):
            return bag_of_contexts
        
        if(len(bag_of_contexts) > BAG_SIZE):
            return random.sample(bag_of_contexts, BAG_SIZE)
        
        if ONE_HOT:
            blank_path = [[0] * (MAX_NODE_LOOKUP_NUM + 1)] * CONTEXT_SIZE
        else:
            blank_path = ([0] * CONTEXT_SIZE)
            
        return ([blank_path] * (BAG_SIZE - len(bag_of_contexts)) + bag_of_contexts)

    
    def to_padded_bag_of_contexts(self):
        bag_of_contexts = self.to_raw_bag_of_contexts()
        padded = self.raw_to_padded(bag_of_contexts)
        if(ONE_HOT):
            padded = np.array(padded)
            padded = padded.reshape(padded.shape[0], padded.shape[1] * padded.shape[2])
            padded = padded.tolist()
        return padded


print("Total commits found", len(RAW_DATA))
DATA_SLICE = [0,1000]


RAW_DATA_SLICE = RAW_DATA.iloc[DATA_SLICE[0]: DATA_SLICE[1]]


TUPLES = [(row['fix_hash'], row['fix_repo']) for i, row in RAW_DATA_SLICE.iterrows()] + [(row['bug_hash'], row['bug_repo']) for i, row in RAW_DATA_SLICE.iterrows()]
TUPLES = list(set(TUPLES))

import multiprocessing
#Creates a lookup dictionary where any commit SHA can be looked up to grab the Commit object with all the data, + bag of paths
COMMIT_DATA_LOOKUP = defaultdict(list)

def _handle_errors(e):
    print("BIG ERROR, WE SHOULD NEVER GET HERE", e)
    return None

def _to_commit_mp(pair):
    try:
        sha = pair[0]
        repo = pair[1]
        #print(repo)
        #raise ValueError("blah")
        
        # print(repo, sha)

        commit = Commit(sha, repo)
        commit._GH_populate_commit_info()  
        commit._generate_bags_of_contexts()
    except Exception as e:
        return str(e)


    return commit

def create_lookup_mp(pairs):
    results = []
    
    pool = multiprocessing.Pool(processes=4)

    print("APPENDING JOBS...")
    for pair in pairs:
        #print(pair)
        result = pool.apply_async(_to_commit_mp, args=(pair,), error_callback=_handle_errors)
        results.append(result)

    print("FINISHED APPENDING JOBS")

    # Wait for all jobs to finish and collect the results
    final_results = {}
    errorList = []
    num_finished = 0
    num_errors = 0
    for i, result in tqdm(enumerate(results), desc="Processing commit", total=len(results)):
        c = result.get()
        if type(c) == str:
            print("error in pool job",i,":", c)
            num_errors += 1
            errorList.append(pairs[i])
        else:
            final_results[c.sha] = c
            
        num_finished += 1   

    pool.close()
    return final_results

COMMIT_DATA_LOOKUP = create_lookup_mp(TUPLES)
print("DONE PROCESING COMMITS")

with open('../data/commit_lookups/supervised_commit_data_lookup' + str(DATA_SLICE[0]) + "-" + str(DATA_SLICE[1]) + '.pickle', 'wb') as file:
    pickle.dump(COMMIT_DATA_LOOKUP, file, protocol=pickle.HIGHEST_PROTOCOL)


# bs = [len(COMMIT_DATA_LOOKUP[x].bag_of_contexts) for x in COMMIT_DATA_LOOKUP]
# bs_nonzero = [x for x in bs if x > 0]
# print("Bag Sizes:")
# print("All:", bs)
# print(f"Non-zero Median: {np.median(bs_nonzero)}, Non-zero Mean: {np.mean(bs_nonzero)}")

# fc = [len(COMMIT_DATA_LOOKUP[x].get_files_changed()) for x in COMMIT_DATA_LOOKUP]
# fc_nonzero = [x for x in fc if x > 0]
# print("Files Changed per commit:")
# print("All:", fc)
# print(f"Non-zero Median: {np.median(fc_nonzero)}, Non-zero Mean: {np.mean(fc_nonzero)}")
