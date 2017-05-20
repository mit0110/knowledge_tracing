"""Creates a split sampling the students to generate smaller sets"""

import argparse
import numpy
import pandas

STUDENT_ID_COLUMN = 'user_id'


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str,
                        help='The path to the file with the csv dataset.')
    parser.add_argument('--output_filename', type=str,
                        help='The path to store the resulting split.')
    parser.add_argument('--sample_size', type=int, default=500,
                        help='Number of students to sample')
    return parser.parse_args()


def main():
    args = parse_arguments()
    dataset = pandas.read_csv(args.filename)

    selected_students = numpy.random.choice(dataset[STUDENT_ID_COLUMN].unique(),
                                            args.sample_size)

    filtered_dataset = dataset.set_index(STUDENT_ID_COLUMN).loc[
        selected_students].reset_index()

    print 'New dataset shape {}'.format(filtered_dataset.shape)
    print 'Discarded rows {}'.format(
        dataset.shape[0] - filtered_dataset.shape[0])
    print 'New number of unique problems {}'.format(
        filtered_dataset.problem_id.unique().shape)

    filtered_dataset.to_csv(args.output_filename)
    print 'All operations completed'


if __name__ == '__main__':
    main()