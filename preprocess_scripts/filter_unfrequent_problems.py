"""Creates a split sampling the students to generate smaller sets"""

import argparse
import pandas

PROBLEM_ID_COLUMN = 'problem_id'


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str,
                        help='The path to the file with the csv dataset.')
    parser.add_argument('--output_filename', type=str,
                        help='The path to store the resulting split.')
    parser.add_argument('--min_frequency', type=int, default=2,
                        help='Minimum number of occurrences of a problem to be'
                             'included in the dataset')
    parser.add_argument('--replace_with_token', action='store_true',
                        help='Replace filtered out problems with a token.')
    parser.add_argument('--use_cols', nargs='+', type=str,
                        help='Use only the given columns in the final result.')
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.use_cols is not None:
        if PROBLEM_ID_COLUMN not in args.use_cols:
            args.use_cols.append(PROBLEM_ID_COLUMN)
        dataset = pandas.read_csv(args.filename, usecols=args.use_cols)
    else:
        dataset = pandas.read_csv(args.filename)

    filtered_dataset = dataset.groupby(PROBLEM_ID_COLUMN).filter(
        lambda x: len(x) >= args.min_frequency)

    print('New dataset shape {}'.format(filtered_dataset.shape))
    print('Discarded rows {}'.format(
        dataset.shape[0] - filtered_dataset.shape[0]))
    print('New number of unique problems {}'.format(
        filtered_dataset.problem_id.unique().shape))

    filtered_dataset.to_csv(args.output_filename)
    print('All operations completed')


if __name__ == '__main__':
    main()
