import git
import networkx as nx
import tree_sitter
import requests
from datetime import datetime
from github import Github
import base64

from vocab import JAVA_LANGUAGE, FILE_FILTERS, ALL_NODE_TYPES, ALL_NODE_INDEXES, MAX_NODE_LOOKUP_NUM

def CommitFactory(
    BAG_SIZE = 256,
    CONTEXT_SIZE = 16,
    USE_FULL_TUPLES = False,
    GH_ACCESS_TOKEN = None
):


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

        def _populate_commit_info(self):
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
            self.bag_of_contexts = self.to_raw_bag_of_contexts()

        ###################################### GIT and GitHub ###################################### 
            
        ################# GIT #################
        
        def get_files_changed(self):
            try:
                repo = git.Repo(self.repo_path)
                commit = repo.commit(self.sha)
                return [diff.a_path for diff in commit.diff(commit.parents[0]) if diff.a_path.endswith(FILE_FILTERS)]
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
                url = f"https://api.github.com/repos/{self.repo_path}/contents/{file_name}?ref={self.sha}"
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
                url = f"https://api.github.com/repos/{self.repo_path}/contents/{file_name}?ref={self.parent}"
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
                if file_name.endswith('.java'):
                    parser.set_language(JAVA_LANGUAGE)
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


        def pad_each_context(self, bag_of_contexts):
            padded_one_hot_paths = []
            for path in bag_of_contexts:
                if len(path) > CONTEXT_SIZE:
                    continue
                padded_path = [0] * max(CONTEXT_SIZE - len(path), 0) + path[-CONTEXT_SIZE:]
                padded_one_hot_paths.append(padded_path)
            return padded_one_hot_paths

        ###################################### Utility ###################################### 
        
        def to_raw_bag_of_contexts(self):
            if(self.use_github):
                files_changed = self.gh_get_files_changed()
            else:
                files_changed = self.get_files_changed()

            if(len(files_changed) > 32):
                files_changed = []

            if(self.use_github):
                sources_at_commit = [self.gh_get_source_at_commit(filename) for filename in files_changed]
                sources_at_parent = [self.gh_get_source_at_parent(filename) for filename in files_changed]

            else:
                sources_at_commit = [self.get_source_at_commit(filename) for filename in files_changed]
                sources_at_parent = [self.get_source_at_parent(filename) for filename in files_changed]

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

            return contexts

        
        def raw_to_padded(self, bag_of_contexts):
            
            bag_of_contexts = self.pad_each_context(bag_of_contexts)

            
            if(len(bag_of_contexts) == BAG_SIZE):
                return bag_of_contexts
            
            if(len(bag_of_contexts) > BAG_SIZE):
                return random.sample(bag_of_contexts, BAG_SIZE)
            
            blank_path = ([0] * CONTEXT_SIZE)
                
            return ([blank_path] * (BAG_SIZE - len(bag_of_contexts)) + bag_of_contexts)

        
        def to_padded_bag_of_contexts(self):
            bag_of_contexts = self.to_raw_bag_of_contexts()
            padded = self.raw_to_padded(bag_of_contexts)
            return padded
    
    return Commit