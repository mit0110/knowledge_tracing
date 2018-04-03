"""Script to extract the student sequences of problems and labels.

The problems are represented with a sequential id starting in 1, using
as identifier the identifier_column argument.

Both labels and instances are represented using the positive or negative
id of the exercise being referenced. If the id is positive, it means the
interaction was successful.

The output is a pickled tuple where the first element are the sequences
as a list of dense vectors, and the second are the labels.
"""

import argparse
import logging
import os
import numpy
import pandas
import sys

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    parser.add_argument('--embedding_info', type=str,
                        help='Filename to store extra information for the '
                             'element embeddings according to their id.')
    return parser.parse_args()


def check_sequence(sequence, labels):
    """Checks if the labels are consistent with the sequences.

    If the label of step t has a positive number, it must be in the same
    position as the one-hot encoding of the exercise identifier in step t+1
    """
    for index, element in enumerate(sequence[1:]):
        label = labels[index]
        assert numpy.abs(element) == numpy.abs(label)


class ProblemEncoder(object):
    def __init__(self, values):
        self.encoding = {}
        self.values = []
        self._last_id = 1
        for value in numpy.unique(values):
            if value not in self.encoding:
                self.encoding[value] = self.last_id
                self.values.append(value)
                self._last_id += 1
        print('{} different problems found'.format(self.last_id))

    def encode(self, sequence):
        encoded_sequence = []
        for value in sequence:
            assert value in self.encoding
            encoded_sequence.append(self.encoding[value])
        return numpy.array(encoded_sequence)

    @property
    def last_id(self):
        return self._last_id


def main():
    args = parse_arguments()
    identifier_column = args.identifier_column
    important_columns = ['order_id', 'user_id', identifier_column, 'correct',
                         'new_skill_id', 'template_id']
    logging.info('Reading csv file.')
    df = pandas.read_csv(args.filename, usecols=important_columns)

    # Remove the duplicate rows generated to store multiple skills
    df.drop_duplicates(subset=['order_id'], keep='first', inplace=True)

    # Train the one hot encoder
    problem_encoder = ProblemEncoder(df[[identifier_column]])

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
        student_sequence = problem_encoder.encode(
            student_df[identifier_column].values)
        assert student_sequence.shape[0] == student_df.shape[0]
        student_results = student_df.correct.values
        student_results[student_results == 0] = -1  # This has -1 or 1 values.

        student_sequence = numpy.multiply(student_sequence, student_results)
        sequences.append(student_sequence)
        # Generate labels
        # The label of a sequence is the outcome of the next interaction. We
        # have to shift the student_results vector one place and add the
        # encoding size as the End Of Sequence label (always positive).
        sequence_labels = numpy.append(student_sequence[1:],
                                       [problem_encoder.last_id])
        assert sequence_labels.shape == student_sequence.shape
        labels.append(sequence_labels)
        if args.check_output:
            check_sequence(student_sequence, sequence_labels)

    logging.info('Saving objects to file')
    utils.pickle_to_file((numpy.array(sequences), numpy.array(labels)),
                         args.output_filename)
    del sequences
    del labels
    if args.embedding_info is None:
        return

    logging.info('Saving embedding information')
    with open(args.embedding_info, 'w') as embedding_meta_file:
        embedding_meta_file.write(
            'Skill\tTemplate\tProblemType\tAssistmentId\n')
        for value in problem_encoder.values:
            value_df = df[df[identifier_column] == value].iloc[0]
            embedding_meta_file.write('{}\t{}\t{}\t{}\n'.format(
                value_df['new_skill_id'], value_df['template_id'],
                value_df['problem_type'], value_df['assistment_id']))

    logging.info('All operations completed')


if __name__ == '__main__':
    main()
