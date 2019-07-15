log_dir="exp.log"
touch $log_dir
python train.py train "Easy_LSTM" >> $log_dir
python train.py train "DNN_LSTM" >> $log_dir
python train.py train "LR"  >> $log_dir
python train.py train "DNN"  >> $log_dir
