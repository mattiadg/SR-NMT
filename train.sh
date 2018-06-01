layer=$1
gpu=$2
python -u train.py -data data.train.pt -save_model path/to/model/SGU_${layer}layers -layer_norm
      -max_grad_norm 1 -layers_enc $layer -layers_dec $layer -dropout 0.1 -gpus $gpu -optim adam -learning_rate 0.0003 -decoder_type SR -encoder_type SR
      -attn_type dot -save_each 30000 -brnn -rnn_size 500 -epochs 1000000 -word_vec_size 500 > path/to/log.out
