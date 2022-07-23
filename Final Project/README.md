# train
bash train.sh baseline.epoch_10 4 4 --num_train_epochs=10

# train RL
bash train.sh baseline.rl_0.1.epoch_10 2 8 --add_rl_loss=0.1 --num_train_epochs=10

# Generate
python3 generate.py --model (model file)