# Run several classifiers with the same seed dataset, to obtain a significance score

DATE=$(date +%y-%m-%d-%H-%M)
RESULTS_DIRECTORY="ongoing-"$DATE
TENSORBOARD_DIR=../results/dkt_lstm/by_problem_id/embeddings/tensorboard/$RESULTS_DIRECTORY
PREDICTIONS_DIR=../results/dkt_lstm/by_problem_id/embeddings/$RESULTS_DIRECTORY
mkdir $TENSORBOARD_DIR
mkdir $PREDICTIONS_DIR
echo "Using dir " $PREDICTIONS_DIR

model="e-lstm"
echo "Model" $model

max_steps=200
echo "Max steps" $max_steps

lstm_units=200
echo "LSTM units" $lstm_units

dropout=0.1
echo "Dropout" $dropout

batch_size=50
echo "Batch size" $batch_size

embedding_size=50
echo "Embedding size" $embedding_size

echo "No pretrained"
echo "No finetune"

python -u ../assistments_2009/run_embedded_dkt.py --filename ../data/assistments2010/processed/id_by_problem/frequency_geq5/sequences_embedded_problems_geq5.p --base_logs_dirname $TENSORBOARD_DIR --test_prediction_dir $PREDICTIONS_DIR --training_epochs 500 --model $model --hidden_layer_size $lstm_units --batch_size $batch_size --log_values 50 --max_num_steps $max_steps --dropout_ratio $dropout --runs 5 --embedding_size $embedding_size --use_prev_state --random_seed 42

rc=$?; if [[ $rc != 0 ]]; then echo "error in training " && exit $rc; fi

mv $TENSORBOARD_DIR ../results/dkt_lstm/by_problem_id/embeddings/tensorboard/$DATE
mv $PREDICTIONS_DIR ../results/dkt_lstm/by_problem_id/embeddings/$DATE
echo "First classifier operations completed"



DATE=$(date +%y-%m-%d-%H-%M)
RESULTS_DIRECTORY="ongoing-"$DATE
TENSORBOARD_DIR=../results/dkt_lstm/by_problem_id/embeddings/tensorboard/$RESULTS_DIRECTORY
PREDICTIONS_DIR=../results/dkt_lstm/by_problem_id/embeddings/$RESULTS_DIRECTORY
mkdir $TENSORBOARD_DIR
mkdir $PREDICTIONS_DIR
echo "Using dir " $PREDICTIONS_DIR

model="e-lstm"
echo "Model" $model

max_steps=200
echo "Max steps" $max_steps

lstm_units=100
echo "LSTM units" $lstm_units

dropout=0.1
echo "Dropout" $dropout

batch_size=50
echo "Batch size" $batch_size

embedding_size=100
echo "Embedding size" $embedding_size

echo "Pretrained"
echo "Finetuned"

python -u ../assistments_2009/run_embedded_dkt.py --filename ../data/assistments2010/processed/id_by_problem/frequency_geq5/sequences_embedded_problems_geq5.p --base_logs_dirname $TENSORBOARD_DIR --test_prediction_dir $PREDICTIONS_DIR --training_epochs 500 --model $model --hidden_layer_size $lstm_units --batch_size $batch_size --log_values 50 --max_num_steps $max_steps --dropout_ratio $dropout --runs 5 --embedding_size $embedding_size --use_prev_state --random_seed 42 --embedding_model ../data/assistments2010/embeddings/word2vec/problem_id_${embedding_size}.model

rc=$?; if [[ $rc != 0 ]]; then echo "error in training " && exit $rc; fi

mv $TENSORBOARD_DIR ../results/dkt_lstm/by_problem_id/embeddings/tensorboard/$DATE
mv $PREDICTIONS_DIR ../results/dkt_lstm/by_problem_id/embeddings/$DATE
echo "Second classifier operations completed"




DATE=$(date +%y-%m-%d-%H-%M)
RESULTS_DIRECTORY="ongoing-"$DATE
TENSORBOARD_DIR=../results/dkt_lstm/by_problem_id/embeddings/tensorboard/$RESULTS_DIRECTORY
PREDICTIONS_DIR=../results/dkt_lstm/by_problem_id/embeddings/$RESULTS_DIRECTORY
mkdir $TENSORBOARD_DIR
mkdir $PREDICTIONS_DIR
echo "Using dir " $PREDICTIONS_DIR

model="e-lstm"
echo "Model" $model

max_steps=300
echo "Max steps" $max_steps

lstm_units=100
echo "LSTM units" $lstm_units

dropout=0.3
echo "Dropout" $dropout

batch_size=50
echo "Batch size" $batch_size

embedding_size=200
echo "Embedding size" $embedding_size

echo "Pretrained"
echo "No finetune"

python -u ../assistments_2009/run_embedded_dkt.py --filename ../data/assistments2010/processed/id_by_problem/frequency_geq5/sequences_embedded_problems_geq5.p --base_logs_dirname $TENSORBOARD_DIR --test_prediction_dir $PREDICTIONS_DIR --training_epochs 500 --model $model --hidden_layer_size $lstm_units --batch_size $batch_size --log_values 50 --max_num_steps $max_steps --dropout_ratio $dropout --runs 5 --embedding_size $embedding_size --use_prev_state --random_seed 42 --embedding_model ../data/assistments2010/embeddings/word2vec/problem_id_${embedding_size}.model --nofinetune

rc=$?; if [[ $rc != 0 ]]; then echo "error in training " && exit $rc; fi

mv $TENSORBOARD_DIR ../results/dkt_lstm/by_problem_id/embeddings/tensorboard/$DATE
mv $PREDICTIONS_DIR ../results/dkt_lstm/by_problem_id/embeddings/$DATE
echo "Thrid classifier operations completed"



DATE=$(date +%y-%m-%d-%H-%M)
RESULTS_DIRECTORY="ongoing-"$DATE
TENSORBOARD_DIR=../results/dkt_lstm/by_problem_id/coembeddings/tensorboard/$RESULTS_DIRECTORY
PREDICTIONS_DIR=../results/dkt_lstm/by_problem_id/coembeddings/$RESULTS_DIRECTORY
mkdir $TENSORBOARD_DIR
mkdir $PREDICTIONS_DIR
echo "Using dir " $PREDICTIONS_DIR

model="co-abs"
echo "Model" $model

max_steps=200
echo "Max steps" $max_steps

lstm_units=200
echo "LSTM units" $lstm_units

dropout=0.5
echo "Dropout" $dropout

batch_size=50
echo "Batch size" $batch_size

echo "Random"
echo "Finetuned"

python -u ../assistments_2009/run_embedded_dkt.py --filename ../data/assistments2010/processed/id_by_problem/frequency_geq5/sequences_embedded_problems_geq5.p --base_logs_dirname $TENSORBOARD_DIR --test_prediction_dir $PREDICTIONS_DIR --training_epochs 500 --model $model --hidden_layer_size $lstm_units --batch_size $batch_size --log_values 50 --max_num_steps $max_steps --dropout_ratio $dropout --runs 5 --embedding_size $lstm_units --use_prev_state --random_seed 42

rc=$?; if [[ $rc != 0 ]]; then echo "error in training " && exit $rc; fi

mv $TENSORBOARD_DIR ../results/dkt_lstm/by_problem_id/coembeddings/tensorboard/$DATE
mv $PREDICTIONS_DIR ../results/dkt_lstm/by_problem_id/coembeddings/$DATE
echo "Fourth classifier operations completed"




DATE=$(date +%y-%m-%d-%H-%M)
RESULTS_DIRECTORY="ongoing-"$DATE
TENSORBOARD_DIR=../results/dkt_lstm/by_problem_id/coembeddings/tensorboard/$RESULTS_DIRECTORY
PREDICTIONS_DIR=../results/dkt_lstm/by_problem_id/coembeddings/$RESULTS_DIRECTORY
mkdir $TENSORBOARD_DIR
mkdir $PREDICTIONS_DIR
echo "Using dir " $PREDICTIONS_DIR

model="co-tanh"
echo "Model" $model

max_steps=100
echo "Max steps" $max_steps

lstm_units=200
echo "LSTM units" $lstm_units

dropout=0.2
echo "Dropout" $dropout

batch_size=30
echo "Batch size" $batch_size

echo "Pretrained"
echo "Finetuned"

python -u ../assistments_2009/run_embedded_dkt.py --filename ../data/assistments2010/processed/id_by_problem/frequency_geq5/sequences_embedded_problems_geq5.p --base_logs_dirname $TENSORBOARD_DIR --test_prediction_dir $PREDICTIONS_DIR --training_epochs 500 --model $model --hidden_layer_size $lstm_units --batch_size $batch_size --log_values 50 --max_num_steps $max_steps --dropout_ratio $dropout --runs 5 --embedding_size $embedding_size --use_prev_state --random_seed 42 --embedding_model ../data/assistments2010/embeddings/word2vec/problem_id_${lstm_units}.model

rc=$?; if [[ $rc != 0 ]]; then echo "error in training " && exit $rc; fi

mv $TENSORBOARD_DIR ../results/dkt_lstm/by_problem_id/coembeddings/tensorboard/$DATE
mv $PREDICTIONS_DIR ../results/dkt_lstm/by_problem_id/coembeddings/$DATE
echo "Fifth classifier operations completed"



DATE=$(date +%y-%m-%d-%H-%M)
RESULTS_DIRECTORY="ongoing-"$DATE
TENSORBOARD_DIR=../results/dkt_lstm/by_problem_id/coembeddings/tensorboard/$RESULTS_DIRECTORY
PREDICTIONS_DIR=../results/dkt_lstm/by_problem_id/coembeddings/$RESULTS_DIRECTORY
mkdir $TENSORBOARD_DIR
mkdir $PREDICTIONS_DIR
echo "Using dir " $PREDICTIONS_DIR

model="co-square"
echo "Model" $model

max_steps=100
echo "Max steps" $max_steps

lstm_units=200
echo "LSTM units" $lstm_units

dropout=0.2
echo "Dropout" $dropout

batch_size=30
echo "Batch size" $batch_size

echo "Pretrained"
echo "No finetuned"

python -u ../assistments_2009/run_embedded_dkt.py --filename ../data/assistments2010/processed/id_by_problem/frequency_geq5/sequences_embedded_problems_geq5.p --base_logs_dirname $TENSORBOARD_DIR --test_prediction_dir $PREDICTIONS_DIR --training_epochs 500 --model $model --hidden_layer_size $lstm_units --batch_size $batch_size --log_values 50 --max_num_steps $max_steps --dropout_ratio $dropout --runs 5 --embedding_size $embedding_size --use_prev_state --random_seed 42 --embedding_model ../data/assistments2010/embeddings/word2vec/problem_id_${lstm_units}.model --nofinetune

rc=$?; if [[ $rc != 0 ]]; then echo "error in training " && exit $rc; fi

mv $TENSORBOARD_DIR ../results/dkt_lstm/by_problem_id/coembeddings/tensorboard/$DATE
mv $PREDICTIONS_DIR ../results/dkt_lstm/by_problem_id/coembeddings/$DATE
echo "Sixth classifier operations completed"