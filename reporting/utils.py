"""
Common reporting functions.
"""
import os


def try_create_dir(root_dir, f):
    if os.path.isdir(os.path.join(root_dir, f)):
        return os.path.join(root_dir, f)
    else:
        os.makedirs(os.path.join(root_dir, f))
        return os.path.join(root_dir, f)
