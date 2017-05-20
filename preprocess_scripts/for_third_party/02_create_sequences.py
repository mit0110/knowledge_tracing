"""Creates the sequence file using a split.

The output are two files with the sequences and labels
    sequence_lenght
    element1 element2 ...
    label1 label2 ...
"""

from __future__ import absolute_import
import argparse
import numpy as np
import pandas
import pickle
import os

from tqdm import tqdm

SEPARATOR = ','

LABEL_COLUMN = 'correct'

TRAIN_SIZE = 0.7

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str,
                        help='The path to the file with the csv dataset.')
    parser.add_argument('--output_dirname', type=str,
                        help='The path to store the sequences objects.')
    parser.add_argument('--split_filename', type=str,
                        help='The path to the file with the student id'
                             'separated for train and test. If not exists, a'
                             'new split will be created.')
    parser.add_argument('--problem_identifier', type=str, default='problem_id',
                        help='The name of the column to use as identifier')
    parser.add_argument('--resequence_id', action='store_true',
                        help='Replace the problem identifier with sequential '
                             'ids.')
    return parser.parse_args()


def analyze_split(split, dataset):
    student_ids = dataset.user_id.unique()
    train_ids = split['train_ids']
    test_ids = split['test_ids']
    train_sequences_size = dataset.set_index('user_id').loc[train_ids].shape[0]
    print 'Students in train {} ({:.4}%), Students in test {}({:.4}%)'.format(
        len(train_ids), len(train_ids) / float(student_ids.shape[0]),
        len(test_ids), len(test_ids) / float(student_ids.shape[0]))
    print 'New train dataset size = ', train_sequences_size
    print 'Real proportion ', train_sequences_size/float(dataset.shape[0])


def create_split(dataset, train_size, test_size):
    print 'Creating new split'
    student_ids = dataset.user_id.unique()
    train_ids = set(np.random.choice(
        student_ids, int(student_ids.shape[0] * train_size), replace=False))
    test_ids = np.array([x for x in student_ids if x not in train_ids])
    return {'train_ids': train_ids, 'test_ids': test_ids}


def write_sequence(student_df, column, file_):
    sequence_string = SEPARATOR.join([str(x) for x in student_df[column]])
    file_.write('{}\n'.format(sequence_string))


def get_sequences(df, id_column, file_):
    # Separate the student sequences
    groups = df.groupby('user_id')
    student_dfs = [groups.get_group(x) for x in groups.groups]
    sequences = []
    labels = []
    for student_df in tqdm(student_dfs):
        # Write the size
        file_.write('{}\n'.format(student_df.shape[0]))
        student_df.sort_values(by='order_id', inplace=True)
        # Write the sequence
        write_sequence(student_df, id_column, file_)
        # Write the labels
        write_sequence(student_df, LABEL_COLUMN, file_)


def main():
    args = parse_arguments()
    dataset = pandas.read_csv(args.filename)
    # Get the split
    try:
        with open(args.split_filename, 'r') as file_:
            split = pickle.load(file_)
    except IOError:
        split = create_split(dataset, TRAIN_SIZE, 1-TRAIN_SIZE)
        with open(args.split_filename, 'w') as file_:
            pickle.dump(split, file_)
    analyze_split(split, dataset)

    # Remove unused columns
    important_columns = ['order_id', 'user_id', LABEL_COLUMN,
                         args.problem_identifier]
    dataset = dataset[important_columns]
    # Remove the duplicate rows generated to store multiple skills
    dataset.drop_duplicates(subset=['order_id'], keep='first', inplace=True)
    if args.resequence_id:
        # Replace identifiers by sequential ids
        dataset.problem_id, problem_id_map = pandas.factorize(
            dataset[args.problem_identifier])

    dataset = dataset.set_index('user_id')
    # Open the output files
    print 'Creating train file'
    with open(os.path.join(args.output_dirname, 'train.txt'), 'w') as file_:
        # Create the sequences
        get_sequences(dataset.loc[split['train_ids']].reset_index(),
                      args.problem_identifier, file_)
    print 'Creating test file'
    with open(os.path.join(args.output_dirname, 'test.txt'), 'w') as file_:
        get_sequences(dataset.loc[split['test_ids']].reset_index(),
                      args.problem_identifier, file_)


if __name__ == '__main__':
    main()
