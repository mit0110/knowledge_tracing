"""Script to join skills of a single interaction"""

import argparse
import pandas


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str,
                        help='The path to the file to replace in csv format.')
    parser.add_argument('--output_filename', type=str,
                        help='The path to store the sequences objects.')
    return parser.parse_args()


def has_duplicate_rows(dataset):
    return dataset.order_id.unique().shape[0] != dataset.shape[0]


def main():
    args = parse_arguments()
    dataset = pandas.read_csv(args.filename)
    assert has_duplicate_rows(dataset)

    print 'Grouping by order_id'
    new_dataset = dataset[['order_id', 'skill_id']].groupby('order_id')
    new_dataset = new_dataset.agg(
        lambda x: ','.join([str(y) for y in x.values]))

    skill_id_map = {}
    new_dataset['new_skill_id'] = 0
    current_max_skill = 1

    print 'Iterating over tuples'
    for order_id, skills, _ in new_dataset.itertuples():
        skill_id = -1
        if skills not in skill_id_map:
            skill_id_map[skills] = current_max_skill
            skill_id = current_max_skill
            current_max_skill += 1
        else:
            skill_id = skill_id_map[skills]
        new_dataset.loc[order_id,['new_skill_id']] = skill_id
    new_dataset.rename(columns={'skill_id': 'joined_skills'}, inplace=True)

    print 'Joining'
    result = new_dataset.join(dataset.set_index('order_id'),
                              how='left').reset_index()
    # Delete duplicate rows
    print 'Removing duplicates'
    result.drop_duplicates(subset=['order_id'], keep='first', inplace=True)
    assert not has_duplicate_rows(result)

    # Save result
    print 'Saving results'
    result.to_csv(args.output_filename)
    print 'All operations completed'


if __name__ == '__main__':
    main()
