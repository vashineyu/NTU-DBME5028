"""utils.py
"""
import h5py


def load_h5df(filepath):
    with open(filepath, "r") as f:
        key = list(f.keys())[0]
        data = list(f.get(key))
    return data

