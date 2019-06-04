# Exploration script for Co-embedded Assistment models with pretrained embeddings

for distance in co-abs co-square co-norm-fixed co-tanh co-sigm; do
    echo "******************* DISTANCE $distance *******************"
    for i in 1 2 3 4 5 6 7; do
        DATE=$(date +%y-%m-%d-%H-%M)
        RESULTS_DIRECTORY="ongoing-"$DATE
        TENSORBOARD_DIR=../results/dkt_lstm/by_problem_id/coembeddings/tensorboard/$RESULTS_DIRECTORY
        PREDICTIONS_DIR=../results/dkt_lstm/by_problem_id/coembeddings/$RESULTS_DIRECTORY
        mkdir $TENSORBOARD_DIR
        mkdir $PREDICTIONS_DIR
        echo "Using dir " $PREDICTIONS_DIR

        echo "******************* EXPLORING SETTING $i *******************"
        MAX_STEPS=(30 50 100 100 200 300)
        rand_max_steps=${MAX_STEPS[$[$RANDOM % ${#MAX_STEPS[@]}]]}
        echo "Max steps" $rand_max_steps

        LSTM_UNITS=(50 100 200)
        rand_lstm_units=${LSTM_UNITS[$[$RANDOM % ${#LSTM_UNITS[@]}]]}
        echo "LSTM units" $rand_lstm_units

        DROPOUT=(0.1 0.2 0.3 0.4 0.5)
        rand_dropout=${DROPOUT[$[$RANDOM % ${#DROPOUT[@]}]]}
        echo "Dropout" $rand_dropout

        BATCH_SIZE=(30 50 50 100 200)
        rand_batch_size=${BATCH_SIZE[$[$RANDOM % ${#BATCH_SIZE[@]}]]}
        echo "Batch size" $rand_batch_size

        echo "Pretrained"

        EPOCHS=300

        CUDA_VISIBLE_DEVICES=0 python -u ../assistments_2009/run_embedded_dkt.py --filename ../data/assistments2010/processed/id_by_problem/frequency_geq5/sequences_embedded_problems_geq5.p --base_logs_dirname $TENSORBOARD_DIR --test_prediction_dir $PREDICTIONS_DIR --training_epochs $EPOCHS --hidden_layer_size $rand_lstm_units --batch_size $rand_batch_size --log_values 50 --max_num_steps $rand_max_steps --dropout_ratio $rand_dropout --runs 1 --use_prev_state --model $distance --embedding_model ../data/assistments2010/embeddings/word2vec/problem_id_${rand_lstm_units}.model 

        mv $TENSORBOARD_DIR ../results/dkt_lstm/by_problem_id/coembeddings/tensorboard/$DATE
        mv $PREDICTIONS_DIR ../results/dkt_lstm/by_problem_id/coembeddings/$DATE
        echo "*******************Exploration finished*******************"
    done 
done
echo "All operations completed"

