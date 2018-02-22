import argparse
import os
import tensorflow as tf
import utils

import assistment_dataset
from models import embedded_dkt
from gensim.models import Word2Vec

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_logs_dirname', type=str, default=None,
                        help='Path to directory to store tensorboard info')
    parser.add_argument('--filename', type=str,
                        help='The path to the pickled file with the processed'
                             'sequences.')
    parser.add_argument('--test_predictions_filename', type=str,
                        help='The path to the file to store the predictions')
    parser.add_argument('--training_epochs', type=int, default=1000,
                        help='The number of epochs to run.')
    parser.add_argument('--hidden_layer_size', type=int, default=100,
                        help='Number of cells in the recurrent layer.')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Number if instances to process at the same time.')
    parser.add_argument('--log_values', type=int, default=50,
                        help='How many training epochs to wait before logging'
                             'the accuracy in validation.')
    parser.add_argument('--max_num_steps', type=int, default=100,
                        help='Number of time steps to unroll the network.')
    parser.add_argument('--embedding_size', type=int, default=100,
                        help='Number of units in the embedding layer.')
    parser.add_argument('--embedding_metadata', type=str, default=None,
                        help='Filename with tsv metadata for embeddings. '
                             'MUST BE AN ABSOLUTE PATH')
    parser.add_argument('--dropout_ratio', type=float, default=0.3,
                        help='Dropout for the input layer and the recurrent '
                             'layer.')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of times to run the experiment with'
                             'different samples')
    parser.add_argument('--use_prev_state', action='store_true',
                        help='Use the ending previous state when processing '
                             'the same instance.')
    parser.add_argument('--nofinetune', action='store_true',
                        help='Do no change the pretrained embedding.')
    parser.add_argument('--embedding_model', type=str, default=None,
                        help='Path to word2vec model to use as pretrained '
                             'embeddings.')
    return parser.parse_args()


def read_configuration(args):
    config = {
        'hidden_layer_size': args.hidden_layer_size,
        'batch_size': args.batch_size,
        'log_values': args.log_values,
        'max_num_steps': args.max_num_steps,
        'embedding_size': args.embedding_size,
        'dropout_ratio': args.dropout_ratio,
        'use_prev_state': args.use_prev_state,
        'finetune_embeddings': not args.nofinetune,
    }
    dataset_config = {'train': 0.7, 'test': 0.2, 'validation': 0.1}
    return config, dataset_config


def read_embedding_model(model_path):
    if model_path is None:
        return None
    return Word2Vec.load(model_path)


def main():
    args = parse_arguments()
    embedding_model = read_embedding_model(args.embedding_model)
    dataset = assistment_dataset.AssistmentDataset(embedding_model)
    sequences, labels = utils.pickle_from_file(args.filename)
    experiment_config, partitions = read_configuration(args)
    print('Creating samples')

    dataset.create_samples(
        sequences, labels, partition_sizes=partitions, samples_num=args.runs)

    for run in range(args.runs):
        print('Running iteration {} of {}'.format(run + 1, args.runs))
        dataset.set_current_sample(run)

        print('Dataset Configuration')
        print(partitions)
        print('Experiment Configuration')
        print(experiment_config)
        if args.base_logs_dirname:
            tf.reset_default_graph()
            logs_dirname = os.path.join(
                args.base_logs_dirname, 'run{}'.format(run))
            utils.safe_mkdir(logs_dirname)
            experiment_config['logs_dirname'] = logs_dirname

        model = embedded_dkt.EmbeddedSeqLSTMModel(
            dataset, embedding_model=embedding_model, **experiment_config)
        model.fit(partition_name='train', close_session=False,
                  training_epochs=args.training_epochs)

        if args.embedding_metadata is not None:
            model.write_embeddings(args.embedding_metadata)
        predicted_labels = model.predict('test')
        utils.pickle_to_file(predicted_labels, os.path.join(
            args.test_predictions_filename, 'predictions_run{}.p'.format(run)))
        utils.pickle_to_file(
            (model.training_performance, model.validation_performance),
            os.path.join(args.test_predictions_filename,
                         'performances_run{}.p'.format(run)))


if __name__ == '__main__':
    main()
