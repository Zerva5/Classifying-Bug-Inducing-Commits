from tree_sitter import Language
import json
import requests

JAVA_LANGUAGE = Language('ast-bindings/build/ast.so', 'java')

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
