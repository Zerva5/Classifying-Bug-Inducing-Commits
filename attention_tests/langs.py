from tree_sitter import Language, Parser

Language.build_library(
  # Store the library in the `build` directory
  'ast-bindings/build/ast.so',

  # Include one or more languages
  [
    'ast-bindings/tree-sitter-c',
    'ast-bindings/tree-sitter-cpp',
    'ast-bindings/tree-sitter-java',
    'ast-bindings/tree-sitter-javascript',
    'ast-bindings/tree-sitter-python'
  ]
)

JS_LANGUAGE = Language('build/ast.so', 'javascript')
PY_LANGUAGE = Language('build/ast.so', 'python')
PY_LANGUAGE = Language('build/ast.so', 'java')
PY_LANGUAGE = Language('build/ast.so', 'c')
PY_LANGUAGE = Language('build/ast.so', 'cpp')

parser = Parser()
parser.set_language(PY_LANGUAGE)
