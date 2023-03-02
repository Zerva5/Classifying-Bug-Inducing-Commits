from process_commit import process_commit
from random_commit import get_random_commit_shas
import ast
import sys

def create_dataset(path, size=32, threshold=50):
    commits = get_random_commit_shas(path, size)
    X_train = [process_commit(commit, path) for commit in commits]
    X_train = [data for data in X_train if len(data) > 0]
    y_train = [(1 if len(x) > threshold else 0) for x in X_train]

    # Determine the input and output dimensions
    try:
        input_dim = len(X_train[0][0])
    except:
        input_dim = 1        
    try:
        output_dim = len(X_train[0][0][0])
    except:
        output_dim = 1        

    all_node_types = []
    for name in dir(ast):
        if not name.startswith('_'):
            attr = getattr(ast, name)
            if isinstance(attr, type) and issubclass(attr, ast.AST):
                all_node_types.append(name)
    all_node_types = [i for i, _ in enumerate(all_node_types)]
    max_num = max(all_node_types)
    X = []
    for num in all_node_types:
        num_one_hot = [0] * (max_num+1)
        num_one_hot[int(num)] = 1
        # X.append(num_one_hot)
        X.append(num)
    # print(X)

    P = [item for sublist in X_train for item in sublist]
    P = [arr for i, arr in enumerate(P) if arr not in P[:i]]

    Y = [0,1]

    d=150

    return X_train, y_train, input_dim, output_dim, X, P, d, Y
