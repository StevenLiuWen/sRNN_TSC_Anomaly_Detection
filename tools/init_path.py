import sys

def add_path(*args):
    for path in args:
        if path not in sys.path:
            sys.path.append(path)
