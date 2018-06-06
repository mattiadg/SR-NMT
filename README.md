# OpenNMT: Open-Source Neural Machine Translation

This is an extension of [OpenNMT](https://github.com/OpenNMT/OpenNMT),
which includes the code for the SR-NMT that has been introduced in 
[Deep Neural Machine Translation with Weakly-Recurrent Units](https://arxiv.org/abs/1805.04185). 

<center style="padding: 40px"><img width="70%" src="https://github.com/mattiadg/SR-NMT/blob/master/docs/img/architecture.png?raw=true" /></center>

## Quickstart

## Some useful tools:

The example below uses the Moses tokenizer (http://www.statmt.org/moses/) to prepare the data and the moses BLEU script for evaluation.

```bash
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/tokenizer.perl
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.de
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.en
sed -i "s/$RealBin\/..\/share\/nonbreaking_prefixes//" tokenizer.perl
wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl
```

## A simple pipeline:

Download and preprocess the data as you would do for [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).
Then use preprocess.py, train.sh and translate.sh for the actual training and translation.

### 1) Preprocess the data.

```bash
python preprocess.py -train_src /path/to/data/train.src -train_tgt /path/to/data/train.tgt -valid_src /path/to/data/valid.src -valid_tgt /path/to/data/valid.tgt -save_data /path/to/data/data
```

### 2) Train the model.

```bash
sh train.sh num_layers num_gpu
```

### 3) Translate sentences.

```bash
sh translate.sh model_name test_file num_gpu
```

### 4) Evaluate.
```bash
sh eval.sh hypothesys target_language /path/to/test/tokenized.tgt
```
This evaluation is consistent with the one used in the paper and was taken from [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/get_ende_bleu.sh).

## New versions
We are working to integrate SR-NMT inside:  
- [OpenNMT-py](https://github.com/mattiadg/OpenNMT-py)  OpenNMT/OpenNMT-py#748

## Citation

If you use this software, please cite:

```
@inproceedings{digangi2018deep,
  author    = {Di Gangi, Mattia A and Federico, Marcello},
  title     = {Deep Neural Machine Translation with Weakly-Recurrent Units},
  booktitle = {Proceedings of the 21st Annual Conference of the European Association for Machine Translation},
  pages     = {119--128},
  year      = {2018}
}
```


[OpenNMT technical report](https://doi.org/10.18653/v1/P17-4012)

```
@inproceedings{opennmt,
  author    = {Guillaume Klein and
               Yoon Kim and
               Yuntian Deng and
               Jean Senellart and
               Alexander M. Rush},
  title     = {OpenNMT: Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proc. ACL},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-4012},
  doi       = {10.18653/v1/P17-4012}
}
```