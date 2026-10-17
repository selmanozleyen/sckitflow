"""Fail on a function signature taking ``**kwargs``, unless its line says ``# allow-kwargs``.

``**kwargs: Unpack[SomeTypedDict]`` passes, since the TypedDict names every key.

Forwarded ``**kwargs`` swallow misspelled settings silently, so every parameter has to be named.
"""

import ast
import sys

MARKER = "# allow-kwargs"


def main(paths: list[str]) -> int:
    """Print each offending signature in ``paths``; return 1 if there was one."""
    failed = 0
    for path in paths:
        with open(path, encoding="utf-8") as f:
            source = f.read()
        lines = source.splitlines()
        for node in ast.walk(ast.parse(source, path)):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) and node.args.kwarg:
                kwarg = node.args.kwarg
                ann = kwarg.annotation
                typed = isinstance(ann, ast.Subscript) and ast.unparse(ann.value).endswith("Unpack")
                if not typed and MARKER not in lines[kwarg.lineno - 1]:
                    print(
                        f"{path}:{kwarg.lineno}: `**{kwarg.arg}` in a signature, name the parameters or add `{MARKER}`"
                    )
                    failed = 1
    return failed


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
