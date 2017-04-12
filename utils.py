"""Helper functions."""

import cPickle
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
    with open(filename, 'w') as file_:
        cPickle.dump(object_, file_, cPickle.HIGHEST_PROTOCOL)


def pickle_from_file(filename):
    """Reads object from filename in cPickle format"""
    with open(filename, 'r') as file_:
        return cPickle.load(file_)
