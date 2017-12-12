"""Helper functions."""

try:
    import cPickle as pickle
except ImportError:
    import pickle

import logging
import os


def safe_mkdir(dir_name):
    """Checks if a directory exists, and if it doesn't, creates one."""
    try:
        os.stat(dir_name)
    except OSError:
        os.mkdir(dir_name)


def pickle_to_file(object_, filename):
    """Dumps object to filename in cPickle format"""
    with open(filename, 'wb') as file_:
        pickle.dump(object_, file_)


def pickle_from_file(filename):
    """Reads object from filename in cPickle format"""
    with open(filename, 'rb') as file_:
        return pickle.load(file_)
