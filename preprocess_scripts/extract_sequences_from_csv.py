"""Script to extract the student sequences of problems and labels.

The problems are represented as described in Piechs DKT implementation, using
as identifier the identifier_column argument.

The output is a pickled tuple where the first element are the sequences
of one-hot representations in sparse format, and the second are the labels
"""

import argparse
import logging
import os
import numpy
import pandas
import sys

from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname( os.path.abspath(__file__))))
import utils
logging.basicConfig(level=logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str,
                        help='The path to the file to replace in csv format.')
    parser.add_argument('--output_filename', type=str,
                        help='The path to store the sequences objects.')
    parser.add_argument('--identifier_column', type=str, default='problem_id',
                        help='The column to use as identifier.')
    parser.add_argument('--check_output', action='store_true',
                        help='Checks if the labels are consistent.')
    return parser.parse_args()


def EOS_vector(classes_num):
    """A representation of the EOS symbol.

    This is the class corresponding to the last element of each sequence."""
    zeros = numpy.zeros(classes_num)
    zeros[-1] = 1
    return zeros

def check_sequence(sequence, labels):
    """Checks if the labels are consistent with the sequences.

    If the label of step t has a positive number, it must be in the same
    position as the one-hot encoding of the exercise identifier in step t+1
    """
    labels = labels.todense()
    for index, element in enumerate(sequence.todense()[1:]):
        label = labels[index]
        if label.max() != 1.0:
            continue
        exercise_index = numpy.argmax(label)
        assert element[0, exercise_index] == 1


def main():
    args = parse_arguments()
    identifier_column = args.identifier_column
    important_columns = ['order_id', 'user_id', identifier_column, 'correct']
    logging.info('Reading csv file.')
    df = pandas.read_csv(args.filename, usecols=important_columns)

    # Remove the duplicate rows generated to store multiple skills
    df.drop_duplicates(subset=['order_id'], keep='first', inplace=True)

    # Train the one hot encoder
    problem_vectorizer = OneHotEncoder()
    problem_vectorizer.fit(df[[identifier_column]])

    # Separate the student sequences
    groups = df.groupby('user_id')
    student_dfs = [groups.get_group(x) for x in groups.groups]
    sequences = []
    labels = []
    eos_vector = EOS_vector(len(problem_vectorizer.active_features_) + 1)
    for student_df in tqdm(student_dfs):
        student_df.sort_values(by='order_id', inplace=True)
        # Generate instances.
        # Each instance is a concatenation of the problem one hot encoding and
        # the one hot encoding vector multiplied point-wise by the 'correct'
        # column.
        student_sequence = problem_vectorizer.transform(
            student_df[[identifier_column]])
        assert student_sequence.indices.shape[0] == student_df.shape[0]
        student_results = student_sequence.multiply(sparse.csr_matrix(
            student_df.correct.values).T)
        sequences.append(sparse.hstack((student_sequence, student_results),
                                       dtype=numpy.int32))
        # Generate labels
        # The label of a sequence is the outcome of the next interaction. We
        # have to shift the student_results vector one place, expand the one-hot
        # encoding size to support the End Of Sequence label (always positive)
        # and add the label for the last element of the sequence.
        sequence_labels = sparse.hstack([
            student_results[1:],
            numpy.zeros((student_results.shape[0] - 1, 1))])
        sequence_labels = sparse.vstack([sequence_labels, eos_vector],
                                        dtype=numpy.int32)
        labels.append(sequence_labels)
        if args.check_output:
            check_sequence(sequences[-1], sequence_labels)

    logging.info('Saving objects to file')
    utils.pickle_to_file((numpy.array(sequences), numpy.array(labels)),
                         args.output_filename)
    logging.info('All operations completed')


if __name__ == '__main__':
    main()
