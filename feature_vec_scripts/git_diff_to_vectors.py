import sys
import ast
import git
from sklearn.preprocessing import OneHotEncoder
import itertools


all_node_types = []
for name in dir(ast):
    if not name.startswith('_'):
        attr = getattr(ast, name)
        if isinstance(attr, type) and issubclass(attr, ast.AST):
            all_node_types.append(name)
type_combinations = list(itertools.product(all_node_types, repeat=2))

def get_file_contents(commit, file_path):
    """
    Return the contents of a file at a specific commit.
    """
    contents = commit.tree[file_path].data_stream.read().decode('utf-8')
    return contents

def get_ast(contents):
    """
    Return the abstract syntax tree for the given file contents.
    """
    tree = ast.parse(contents)
    return tree

def get_paths(tree):
    """
    Return a set of all unique paths in the given abstract syntax tree.
    """
    paths = set()
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            paths.add((type(node).__name__, type(child).__name__))
    return paths

if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Usage: python git_diff_to_vectors.py <commit_sha> <repo_path>")
        sys.exit(1)


    commit_sha = sys.argv[1]
    repo_path = sys.argv[2]

    print(f"Fetching repo at {repo_path}")
    repo = git.Repo(repo_path)
    print(f"Fetched repo at {repo_path}")

    commit = repo.commit(commit_sha)
    print(f"Got commit {commit_sha}")

    changed_py_files = [diff.a_path for diff in commit.diff(commit.parents[0]) if diff.a_path.endswith('.py')]

    print("Changed .py files in commit:")
    for file in changed_py_files:
        print(file)

    pre_commit_trees = []
    post_commit_trees = []

    for file_path in changed_py_files:
        print(f"Getting file contents for {file_path} pre-commit")
        pre_commit_contents = get_file_contents(commit.parents[0], file_path)
        print(f"Got file contents for {file_path} pre-commit")

        print(f"Building AST for {file_path} pre-commit")
        pre_commit_tree = get_ast(pre_commit_contents)
        pre_commit_trees.append(pre_commit_tree)
        print(f"Built AST for {file_path} pre-commit")

        print(f"Getting file contents for {file_path} post-commit")
        post_commit_contents = get_file_contents(commit, file_path)
        print(f"Got file contents for {file_path} post-commit")

        print(f"Building AST for {file_path} post-commit")
        post_commit_tree = get_ast(post_commit_contents)
        post_commit_trees.append(post_commit_tree)
        print(f"Built AST for {file_path} post-commit")

    pre_commit_paths = set()
    for tree in pre_commit_trees:
        pre_commit_paths |= get_paths(tree)

    post_commit_paths = set()
    for tree in post_commit_trees:
        post_commit_paths |= get_paths(tree)

    unique_paths = pre_commit_paths.symmetric_difference(post_commit_paths)

    print("Unique paths:")
    for path in unique_paths:
        print(path)

    encoder = OneHotEncoder(categories=[all_node_types, all_node_types])
    encoded_paths = encoder.fit_transform(list(unique_paths))

    print("Encoded paths:")
    print(encoded_paths.toarray())
