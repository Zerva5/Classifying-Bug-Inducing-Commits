import sys
import ast
import git
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MultiLabelBinarizer
import numpy as np

def process_commit(commit_sha, repo_path):
    def get_file_contents(commit, file_path):
        return commit.tree[file_path].data_stream.read().decode('utf-8')

    def get_ast(contents):
        return ast.parse(contents)

    def get_paths(tree):
        paths = set()

        # Recursive function to explore the tree
        def explore(node, path):
            # Add current node to path
            path.append(type(node).__name__)

            # If the node has no children, it's a leaf node and the path is complete
            if not list(ast.iter_child_nodes(node)):
                paths.add(tuple(path))
            else:
                # Explore each child node recursively
                for child in ast.iter_child_nodes(node):
                    explore(child, path)

            # Remove current node from path before returning
            path.pop()

        # Start exploring from the root node
        root = ast.parse("")
        explore(tree, [])

        return paths

    ########################################## Processing and setup ##########################################

    repo = git.Repo(repo_path)
    commit = repo.commit(commit_sha)
    changed_py_files = [diff.a_path for diff in commit.diff(commit.parents[0]) if diff.a_path.endswith('.py')]

    ########################################## Compute Abstract Syntax Trees ##########################################

    pre_commit_trees = []
    post_commit_trees = []

    for file_path in changed_py_files:
        print(commit.parents[0], file_path)
        try:
            pre_commit_contents = get_file_contents(commit.parents[0], file_path)
            pre_commit_tree = get_ast(pre_commit_contents)
            pre_commit_trees.append(pre_commit_tree)
            post_commit_contents = get_file_contents(commit, file_path)
            post_commit_tree = get_ast(post_commit_contents)
            post_commit_trees.append(post_commit_tree)
        except:
            pass

    ########################################## Compute Bag of Contexts (paths) ##########################################


    pre_commit_paths = set()
    for tree in pre_commit_trees:
        pre_commit_paths |= get_paths(tree)

    post_commit_paths = set()
    for tree in post_commit_trees:
        post_commit_paths |= get_paths(tree)

    unique_paths = pre_commit_paths.symmetric_difference(post_commit_paths)

    ########################################## Map Symbols to a number/index/id ##########################################

    all_node_types = []
    for name in dir(ast):
        if not name.startswith('_'):
            attr = getattr(ast, name)
            if isinstance(attr, type) and issubclass(attr, ast.AST):
                all_node_types.append(name)

    max_num = len(all_node_types)

    mapped_paths = []
    for path in unique_paths:
        mapped_path = []
        for node in path:
            index = all_node_types.index(node)
            mapped_path.append(index + 1)
        mapped_paths.append(mapped_path)

    ########################################## Convert to One-Hot Encoding ##########################################

    one_hot_paths = []
    
    # Iterate over each row in the array
    for row in mapped_paths:
        # Create an empty list to hold the one-hot encodings for this row
        row_one_hot = []
        
        # Iterate over each element in the row
        for num in row:
            # Create an empty list to hold the one-hot encoding for this number
            num_one_hot = [0] * (max_num+1)
            
            # Set the corresponding element to 1
            num_one_hot[int(num)] = 1
            
            # Add the one-hot encoding for this number to the row's list
            row_one_hot.append(num_one_hot)
        
        # Add the row's list of one-hot encodings to the main list
        one_hot_paths.append(row_one_hot)


    ########################################## Pad to a fixed length ##########################################

    padded_one_hot_paths = []
    
    SET_PATH_LENGTH = 32

    for path in one_hot_paths:
        padded_path = [[0] * (max_num+1)] * max(SET_PATH_LENGTH - len(path), 0) + path[-SET_PATH_LENGTH:]
        padded_one_hot_paths.append(padded_path)
    
    return padded_one_hot_paths