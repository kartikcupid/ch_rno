#!/usr/bin/env python
"""Quick static checks for the project's Python files.

    python3 check_code.py            check every project .py file
    python3 check_code.py a.py b.py  check only these

Checks: the file compiles; no line is longer than 88 characters; f-strings
parse on Python 3.11 (the cluster venv), i.e. no quote character of an
enclosing f-string reused inside a replacement field, no backslash inside a
replacement field, no comment inside one; and rno_train.py / rno_predict.py
carry no comments (docstrings are fine).  Exit status 1 on any problem.
"""

from __future__ import annotations

import glob
import io
import os
import sys
import tokenize

MAX_LEN = 88
NO_COMMENTS = {"rno_train.py", "rno_predict.py"}
FSTART = getattr(tokenize, "FSTRING_START", None)
FMIDDLE = getattr(tokenize, "FSTRING_MIDDLE", None)
FEND = getattr(tokenize, "FSTRING_END", None)


def quote_of(tok):
    s = tok.lstrip("rRbBuUfF")
    return s[:3] if s[:3] in ('"""', "'''") else s[:1]


def check_fstrings(path, src):
    """Flag f-string constructs that Python 3.12 accepts but 3.11 rejects."""
    out, stack = [], []
    toks = tokenize.generate_tokens(io.StringIO(src).readline)
    for tok in toks:
        kind, text, (row, _), _, _ = tok
        inside = bool(stack)
        if inside and kind in (tokenize.STRING, FSTART):
            q = quote_of(text)
            if any(q[0] == o[0] for o in stack):
                out.append(f"{path}:{row}: quote {q} reused inside an f-string")
            if kind == tokenize.STRING and "\\" in text:
                out.append(f"{path}:{row}: backslash inside an f-string field")
        if inside and kind == tokenize.COMMENT:
            out.append(f"{path}:{row}: comment inside an f-string field")
        if kind == FSTART:
            stack.append(quote_of(text))
        elif kind == FEND and stack:
            stack.pop()
    return out


def check_file(path):
    probs = []
    with open(path, encoding="utf-8") as f:
        src = f.read()
    try:
        compile(src, path, "exec")
    except SyntaxError as e:
        return [f"{path}:{e.lineno}: does not compile: {e.msg}"]
    for i, line in enumerate(src.splitlines(), 1):
        if len(line) > MAX_LEN:
            probs.append(f"{path}:{i}: line is {len(line)} > {MAX_LEN} chars")
    if FSTART is not None:
        probs += check_fstrings(path, src)
    if os.path.basename(path) in NO_COMMENTS:
        toks = tokenize.generate_tokens(io.StringIO(src).readline)
        for tok in toks:
            if (tok.type == tokenize.COMMENT and "noqa" not in tok.string
                    and not tok.string.startswith("#!")):
                probs.append(f"{path}:{tok.start[0]}: comment in {path}")
    return probs


def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    files = argv or sorted(glob.glob(os.path.join(here, "*.py"))
                           + glob.glob(os.path.join(here, "tests", "*.py")))
    probs = [p for f in files for p in check_file(f)]
    for p in probs:
        print(p)
    print(f"checked {len(files)} file(s): "
          f"{'OK' if not probs else f'{len(probs)} problem(s)'}")
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
