"""Source-defined function contracts, independent of runtime test guards."""
import ast
from pathlib import Path


def function_text(path,name):
    source=Path(path).read_text();nodes=[n for n in ast.parse(source).body
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name]
    if len(nodes)!=1:raise ValueError('one explicit source observation function required: '+name)
    node=nodes[0];first=min([node.lineno,*[d.lineno for d in node.decorator_list]])
    return ''.join(source.splitlines(keepends=True)[first-1:node.end_lineno])
