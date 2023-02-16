import sys
import os
import ast
import git
import numpy as np

# Define helper functions to extract method names and create abstract syntax trees
def extract_method_names(node):
    method_names = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.FunctionDef):
            method_names.append(child.name)
        else:
            method_names.extend(extract_method_names(child))
    return method_names

def create_ast(code):
    try:
        return ast.parse(code)
    except SyntaxError:
        return None

# Parse command line arguments
if len(sys.argv) != 3:
    print("Usage: python script.py <commit-sha> <path-to-repo>")
    sys.exit(1)
commit_sha = sys.argv[1]
repo_path = sys.argv[2]

# Clone the repository to a temporary directory
temp_dir = os.path.join(os.getcwd(), 'temp')
if os.path.isdir(temp_dir):
    os.system(f'rm -rf {temp_dir}')
os.makedirs(temp_dir)
repo = git.Repo.clone_from(repo_path, temp_dir)

# Get the commit and its parent
commit = repo.commit(commit_sha)
parent_commit = commit.parents[0] if commit.parents else None

# Initialize a dictionary to store method snapshots
snapshots = {}

# Iterate over all changed files in the commit
for diff in commit.diff(parent_commit):
    # Ignore binary files and files that don't end with .py
    if diff.a_blob is None or not diff.a_blob.path.endswith('.py'):
        continue

    # Get the file contents before and after the commit
    before_content = diff.a_blob.data_stream.read().decode()
    after_content = diff.b_blob.data_stream.read().decode()

    # Create abstract syntax trees for both versions of the file
    before_ast = create_ast(before_content)
    after_ast = create_ast(after_content)

    # Extract the names of all methods that have changed
    if before_ast and after_ast:
        before_names = set(extract_method_names(before_ast))
        after_names = set(extract_method_names(after_ast))
        changed_names = before_names.symmetric_difference(after_names)

        # Take a snapshot of each changed method
        for name in changed_names:
            if name not in snapshots:
                snapshots[name] = []
            snapshots[name].append(after_content)

# Compute the set of unique paths between each terminal of each tree
contexts = set()
for snapshot in snapshots.values():
    for code in snapshot:
        tree = create_ast(code)
        if tree:
            for node in ast.walk(tree):
                if isinstance(node, ast.Expr):
                    context = []
                    while node:
                        if isinstance(node, ast.FunctionDef):
                            context.append(node.name)
                        elif isinstance(node, ast.ClassDef):
                            context.append(node.name)
                        elif isinstance(node, ast.Module):
                            context.append('<module>')
                        node = getattr(node, 'parent', None)
                    contexts.add(tuple(context[::-1]))

# Encode the set of contexts as a set of vectors using one-hot encoding
unique_contexts = sorted(list(contexts))
num_contexts = len(unique_contexts)
context_vectors = np.zeros((num_contexts, num_contexts))
for i, context in enumerate(unique_contexts):
    for j, other_context in enumerate(unique_contexts):
        if i != j and context[:-1] == other_context[:-1]:
            context_vectors[i, j] = 1

# Print the resulting context vectors
print(context_vectors)
