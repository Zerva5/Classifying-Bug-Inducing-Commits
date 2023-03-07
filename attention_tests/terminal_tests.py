import sys
import ast
import git
import numpy as np
import random
from tqdm.auto import tqdm, trange
import os
from datetime import datetime
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, ALL_COMPLETED

import networkx as nx
import matplotlib.pyplot as plt

ALL_NODE_TYPES = []
ALL_NODE_INDEXES = []
i = 0
for name in dir(ast):
    if not name.startswith('_'):
        attr = getattr(ast, name)
        if isinstance(attr, type) and issubclass(attr, ast.AST):
            ALL_NODE_TYPES.append(name)
            ALL_NODE_INDEXES.append(i)
            i += 1

MAX_NODE_LOOKUP_NUM = len(ALL_NODE_TYPES)

class Commit:
    def __init__(self, sha, repo_path):
        self.sha = sha
        self.repo_path = repo_path
        self.parent = None
        self.author = None
        self.datetime = None
        self.message = None
        self._populate_commit_info()

    def _populate_commit_info(self):
        repo = git.Repo(self.repo_path)
        commit = repo.commit(self.sha)
        self.parent = commit.parents[0].hexsha if commit.parents else None
        self.author = commit.author.name if commit.author else None
        self.datetime = datetime.fromtimestamp(commit.committed_date)
        self.message = commit.message if commit.message else None

    def get_testing_y_label(self):
        return 1 if ("fix" in self.message) else 0
        
    def get_files_changed(self):
        try:
            repo = git.Repo(self.repo_path)
            commit = repo.commit(self.sha)
            return [diff.a_path for diff in commit.diff(commit.parents[0])][:8] #TODO: Remove this max file count for real dataset
        except:
            return []

    def get_source_at_commit(self, file_name):
        try:
            repo = git.Repo(self.repo_path)
            commit = repo.commit(self.sha)
            return commit.tree[file_name].data_stream.read().decode('utf-8')
        except:
            return ''

    def get_source_at_parent(self, file_name):
        try:
            repo = git.Repo(self.repo_path)
            commit = repo.commit(self.sha)
            return commit.parents[0].tree[file_name].data_stream.read().decode('utf-8')
        except:
            return ''

    def source_to_ast(self, source):
        try:
            return ast.parse(source)
        except:
            return None
        
    def get_root_paths(self, tree):
        try:
            paths = set()

            # Recursive function to explore the tree
            def explore(node, path, terminalA=None):
                # Add current node to path
                if terminalA is None:
                    terminalA = type(node).__name__
                path.append(type(node).__name__)

                # If the node has no children, it's a leaf node and the path is complete
                if not list(ast.iter_child_nodes(node)):
                    paths.add((terminalA, tuple(path), type(node).__name__))
                else:
                    # Explore each child node recursively
                    for child in ast.iter_child_nodes(node):
                        explore(child, path, terminalA)

                # Remove current node from path before returning
                path.pop()

            # Start exploring from the root node
            root = ast.parse("")
            explore(tree, [])
            
            return paths
        except:
            return set([])

    def get_paths(self, node, max_depth=15):
        graph = nx.Graph()
        leaves = []
        for n in ast.walk(node):
            graph.add_node(n)
            if list(ast.iter_child_nodes(n)):
                for child in ast.iter_child_nodes(n):
                    graph.add_edge(n, child)
            else:
                leaves.append(n)

        all_pairs_shortest_paths = dict(nx.all_pairs_shortest_path(graph, cutoff=max_depth))

        paths = set()
        for terminalA in leaves:
            for terminalB in leaves:
                if terminalA == terminalB:
                    continue
                if terminalA not in all_pairs_shortest_paths or terminalB not in all_pairs_shortest_paths[terminalA]:
                    continue
                path = all_pairs_shortest_paths[terminalA][terminalB]
                if path[0] == terminalA and path[-1] == terminalB:
                    paths.add((terminalA.body, tuple([type(n).__name__ for n in path]), terminalB.body))

        return paths



    def ast_to_bag_of_contexts(self, ast_trees, only_roots=False):
        paths = set()
        for tree in ast_trees:
            if(only_roots):
                paths |= self.get_root_paths(tree)
            else:
                paths |= self.get_paths(tree)
        return paths

# commit = Commit('876776e291', '/home/brennan/core')
commit = Commit('0c042e8f72', '/home/brennan/core')

files_changed = commit.get_files_changed()[-1:]
sources_at_commit = [commit.get_source_at_commit(filename) for filename in files_changed]
sources_at_parent = [commit.get_source_at_parent(filename) for filename in files_changed]        

asts_commit = [commit.source_to_ast(source) for source in sources_at_commit]
asts_parent = [commit.source_to_ast(source) for source in sources_at_parent]


print(commit.ast_to_bag_of_contexts(asts_commit, only_roots=False))
# contexts_parent = commit.ast_to_bag_of_contexts(asts_parent, only_roots=True)
