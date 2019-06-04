# Exploration script for Co-embedded Assistment models with pretrained embeddings

for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    DATE=$(date +%y-%m-%d-%H-%M)
    RESULTS_DIRECTORY="ongoing-"$DATE
    TENSORBOARD_DIR=../results/dkt_lstm/by_problem_id/embeddings/tensorboard/$RESULTS_DIRECTORY
    PREDICTIONS_DIR=../results/dkt_lstm/by_problem_id/embeddings/$RESULTS_DIRECTORY
    mkdir $TENSORBOARD_DIR
    mkdir $PREDICTIONS_DIR
    echo "Using dir " $PREDICTIONS_DIR

    echo "******************* EXPLORING SETTING $i *******************"
    MAX_STEPS=(30 50 100 100 200 300)
    rand_max_steps=${MAX_STEPS[$[$RANDOM % ${#MAX_STEPS[@]}]]}
    echo "Max steps" $rand_max_steps

    LSTM_UNITS=(30 50 100 200)
    rand_lstm_units=${LSTM_UNITS[$[$RANDOM % ${#LSTM_UNITS[@]}]]}
    echo "LSTM units" $rand_lstm_units

    DROPOUT=(0.1 0.2 0.3 0.4 0.5)
    rand_dropout=${DROPOUT[$[$RANDOM % ${#DROPOUT[@]}]]}
    echo "Dropout" $rand_dropout

    BATCH_SIZE=(30 50 50 100 200)
    rand_batch_size=${BATCH_SIZE[$[$RANDOM % ${#BATCH_SIZE[@]}]]}
    echo "Batch size" $rand_batch_size

    EMBEDDING_SIZE=(50 100 200)
    rand_embd_size=${EMBEDDING_SIZE[$[$RANDOM % ${#EMBEDDING_SIZE[@]}]]}
    echo "Embedding size" $rand_embd_size

    echo "Pretrained"
    echo "No finetuned"

    EPOCHS=300

    CUDA_VISIBLE_DEVICES=0 python -u ../assistments_2009/run_embedded_dkt.py --filename ../data/assistments2010/processed/id_by_problem/frequency_geq5/sequences_embedded_problems_geq5.p --model e-lstm --base_logs_dirname $TENSORBOARD_DIR --test_prediction_dir $PREDICTIONS_DIR --training_epochs $EPOCHS --hidden_layer_size $rand_lstm_units --batch_size $rand_batch_size --log_values 50 --max_num_steps $rand_max_steps --dropout_ratio $rand_dropout --embedding_size $rand_embd_size --runs 1 --use_prev_state --embedding_model ../data/assistments2010/embeddings/word2vec/problem_id_${rand_embd_size}.model --nofinetune

    mv $TENSORBOARD_DIR ../results/dkt_lstm/by_problem_id/embeddings/tensorboard/$DATE
    mv $PREDICTIONS_DIR ../results/dkt_lstm/by_problem_id/embeddings/$DATE
    echo "*******************Exploration finished*******************"
done
echo "All operations completed"
