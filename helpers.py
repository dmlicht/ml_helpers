from sys import getsizeof
from math import ceil
import pandas as pd

CHUNKSIZE = 10000.0


def print_size(obj):
    """Helper to show how large an object is"""
    size_in_gb = bytes_to_gb(getsizeof(obj))
    print("obj size: %s GB." % size_in_gb)


def bytes_to_gb(n_bytes):
    """Returns n gigabytes in argument bytes"""
    return n_bytes * 1e-9


def count_rows(filename):
    """Returns number or rows in a file"""
    return sum(1 for _ in open(filename)) - 1


def calc_chunks(n_rows, chunksize):
    """Returns number of chunks for given rows and chunksize"""
    return ceil(n_rows / chunksize)


def feature_presence(df):
    """Return a DataFrame representing if features are present or not"""
    return df.isnull().applymap(lambda x: {
        True: 0,
        False: 1
    }[x])


# This style might go away when I learn how to use Dask
def read_csv_sparse(filename, chunksize=CHUNKSIZE):
    """Return a SparseDataFrame read in chunk by chunk"""
    # For Debugging
    n_numeric_rows = count_rows(filename)
    n_chunks = calc_chunks(n_numeric_rows, chunksize)

    partial_frame = None
    numeric_chunks = pd.read_csv(filename, iterator=True, low_memory=False, chunksize=chunksize)
    ii = 0
    for chunk in numeric_chunks:
        ii += 1
        sparse_chunk = chunk.to_sparse()
        if partial_frame is None:
            partial_frame = sparse_chunk
        else:
            partial_frame = partial_frame.append(sparse_chunk)

        # For debugging
        print("chunk " + str(ii) + " of " + str(n_chunks))
        print_size(partial_frame)
    return partial_frame
