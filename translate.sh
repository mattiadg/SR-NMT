model=$1
test=$2
gpu=$3
python translate.py -src $test -model $model -output $model.test.out -gpu $gpu -batch_size 1
