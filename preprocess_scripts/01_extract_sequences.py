"""Script to extract the student sequences of problems and labels.
"""

import argparse
import logging
import os
import numpy
import pandas
import sys

from scipy.sparse import csr_matrix, hstack
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
    return parser.parse_args()


def main():
    args = parse_arguments()
    logging.info('Reading csv file.')
    df = pandas.read_csv(args.filename)
    # Remove unused columns
    important_columns = ['order_id', 'user_id', 'problem_id', 'correct']
    df = df[important_columns]

    # Remove the duplicate rows generated to store multiple skills
    df.drop_duplicates(subset=['order_id'], keep='first', inplace=True)

    # Train the one hot encoder
    problem_vectorizer = OneHotEncoder()
    problem_vectorizer.fit(df[['problem_id']])

    # Separate the student sequences
    groups = df.groupby('user_id')
    student_dfs = [groups.get_group(x) for x in groups.groups]
    sequences = []
    labels = []
    for student_df in tqdm(student_dfs):
        student_df.sort_values(by='order_id', inplace=True)
        # Generate instances.
        # Each instance is a concatenation of the problem one hot encoding and
        # the one hot encoding vector multiplied point-wise by the 'correct'
        # column.
        student_sequence = problem_vectorizer.transform(
            student_df[['problem_id']])
        student_results = student_sequence.multiply(csr_matrix(
            student_df.correct.values).T)
        sequences.append(hstack((student_sequence, student_results)))
        # Generate labels
        # The label of a sequence is a pair with the next exercise one-hot
        # encoding index and the output of the next exercise.
        assert student_sequence.indices.shape[0] == student_df.shape[0]
        labels.append((
            numpy.roll(student_sequence.indices, -1),
            numpy.roll(student_df.correct.values, -1)))

    logging.info('Saving objects to file')
    utils.pickle_to_file((sequences, labels), args.output_filename)
    logging.info('All operations completed')


if __name__ == '__main__':
    main()
