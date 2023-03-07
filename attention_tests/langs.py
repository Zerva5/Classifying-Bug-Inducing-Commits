from tree_sitter import Language, Parser

Language.build_library(
  # Store the library in the `build` directory
  'ast-bindings/build/my-languages.so',

  # Include one or more languages
  [
    'ast-bindings/tree-sitter-c',
    'ast-bindings/tree-sitter-cpp',
    'ast-bindings/tree-sitter-java',
    'ast-bindings/tree-sitter-javascript',
    'ast-bindings/tree-sitter-python'
  ]
)

JS_LANGUAGE = Language('build/ast-langs.so', 'javascript')
PY_LANGUAGE = Language('build/ast-langs.so', 'python')
PY_LANGUAGE = Language('build/ast-langs.so', 'java')
PY_LANGUAGE = Language('build/ast-langs.so', 'c')
PY_LANGUAGE = Language('build/ast-langs.so', 'cpp')

parser = Parser()
parser.set_language(PY_LANGUAGE)
