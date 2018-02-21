"""Script to train word2vec models over sequences.

If gensim is not loading, try with
LD_PRELOAD=~/miniconda3/envs/env_edm2/lib/libmkl_core.so:~/miniconda3/envs/env_edm2/lib/libmkl_sequential.so

"""

import argparse
import os
import numpy
import pickle

from gensim.models import Word2Vec


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filename', type=str,
                        help='The path to the pickled file with sequences.')
    parser.add_argument('--output_filename', type=str,
                        help='The path to store the obtained model.')
    parser.add_argument('--embedding_size', type=int, default=100,
                        help='The size of the output embedding.')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of times to iterate over the dataset.')
    return parser.parse_args()


def main():
    args = parse_arguments()
    with open(args.input_filename, 'rb') as sequence_file:
        raw_sequences, _ = pickle.load(sequence_file)
    import ipdb; ipdb.set_trace()
    # Use only first and second elements
    sequences = [numpy.abs(sequence).astype(numpy.str).tolist()
                 for sequence in raw_sequences]
    print('Processing {} sequences'.format(len(sequences)))

    model_config = {
        "size": args.embedding_size,
        "window": 5,
        "workers": 6,
        "min_count": 5,
        "alpha": 0.01,  # initial learning rate
        "iter": args.iterations,
        "negative": 5,
        "sg": 1,  # Use skipgram model
        # "hs": 1 if args.hs else 0,
        # "cbow_mean": 1 if args.cbow_mean else 0
    }

    print('Start fitting with config {}'.format(model_config))
    model = Word2Vec(sequences, **model_config)
    print('Saving model')
    model.save(args.output_filename)
    print('All iterations completed')


if __name__ == '__main__':
    main()
