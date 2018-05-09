from __future__ import division

import onmt
import onmt.Markdown
import onmt.Models
import onmt.Decoders
import onmt.Encoders
import onmt.modules
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
from torch.nn import init
import math
import time

parser = argparse.ArgumentParser(description='train.py')
onmt.Markdown.add_md_help_argument(parser)

# Data options

parser.add_argument('-data', required=True,
                    help='Path to the *-train.pt file from preprocess.py')
parser.add_argument('-save_model', default='model',
                    help="""Model filename (the model will be saved as
                    <save_model>_epochN_PPL.pt where PPL is the
                    validation perplexity""")
parser.add_argument('-train_from_state_dict', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model's state_dict.""")
parser.add_argument('-train_from', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model.""")

# Model options

parser.add_argument('-model_type', type=str, default='nmt',
                    choices=['nmt', 'lm'],
                    help="""Kind of model to train, it can be
                     neural machine translation or language model
                     [nmt|lm]""")
parser.add_argument('-layers_enc', type=int, default=2,
                    help='Number of layers in the LSTM encoder')
parser.add_argument('-layers_dec', type=int, default=2,
                    help='Number of layers in the LSTM decoder')
parser.add_argument('-rnn_size', type=int, default=500,
                    help='Size of LSTM hidden states')
parser.add_argument('-word_vec_size', type=int, default=500,
                    help='Word embedding sizes')
parser.add_argument('-input_feed', type=int, default=1,
                    help="""Feed the context vector at each time step as
                    additional input (via concatenation with the word
                    embeddings) to the decoder.""")
parser.add_argument('-rnn_type', type=str, default='LSTM',
                    choices=['LSTM', 'GRU', 'SRU'],
                    help="""The gate type to use in the RNNs""")
parser.add_argument('-rnn_encoder_type', type=str,
                    choices=['LSTM', 'GRU', 'SRU'],
                    help="""The gate type to use in the encoder RNNs. It overwrites -rnn_type""")
parser.add_argument('-rnn_decoder_type', type=str,
                    choices=['LSTM', 'GRU', 'SRU'],
                    help="""The gate type to use in the decoder RNNs. It overwrites -rnn_type""")
parser.add_argument('-attn_type', type=str, default='mlp',
                    choices=['mlp', 'dot'],
                    help="""The attention type to use in the decoder""")
parser.add_argument('-activ', type=str, default='tanh',
                     help="""Activation function inside the RNNs.""")
parser.add_argument('-brnn', action='store_true',
                    help='Use a bidirectional encoder')
parser.add_argument('-context_gate', type=str, default=None,
                    choices=['source', 'target', 'both'],
                    help="""Type of context gate to use [source|target|both].
                    Do not select for no context gate.""")
parser.add_argument('-decoder_type', type=str, default='StackedRNN',
                    help="""Decoder neural architecture to use""")
parser.add_argument('-encoder_type', type=str, default='RNN',
                    help="""Encoder architecture""")
parser.add_argument('-layer_norm', default=False, action="store_true",
                    help="""Add layer normalization in recurrent units""")

# Optimization options

parser.add_argument('-batch_size', type=int, default=64,
                    help='Maximum batch size')
parser.add_argument('-max_generator_batches', type=int, default=32,
                    help="""Maximum batches of words in a sequence to run
                    the generator on in parallel. Higher is faster, but uses
                    more memory.""")
parser.add_argument('-epochs', type=int, default=13,
                    help='Number of training epochs')
parser.add_argument('-start_epoch', type=int, default=1,
                    help='The epoch from which to start')
parser.add_argument('-param_init', type=float, default=0.1,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init)""")
parser.add_argument('-optim', default='sgd',
                    help="Optimization method. [sgd|adagrad|adadelta|adam]")
parser.add_argument('-max_grad_norm', type=float, default=5,
                    help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument('-dropout', type=float, default=0.3,
                    help='Dropout probability; applied between LSTM stacks.')
parser.add_argument('-curriculum', action="store_true",
                    help="""For this many epochs, order the minibatches based
                    on source sequence length. Sometimes setting this to 1 will
                    increase convergence speed.""")
parser.add_argument('-extra_shuffle', action="store_true",
                    help="""By default only shuffle mini-batch order; when true,
                    shuffle and re-assign mini-batches""")
parser.add_argument('-change_optimizer', default=False, action='store_true',
                    help="""In case a model is reloaded, it sets the optimizer
                     values to the one set in the arguments""")
parser.add_argument('-enc_short_path', type=bool, default=False,
                    help="""If True, creates a short path from the source embeddings to the output
                    by adding them to the attention""")
parser.add_argument('-use_learning_rate_decay', action="store_true",
                    help='if set, activate learning rate decay after every checkpoint')
parser.add_argument('-save_each', type=int, default=10000,
                    help="""The number of minibatches to compute before saving a checkpoint""")

# learning rate
parser.add_argument('-learning_rate', type=float, default=1.0,
                    help="""Starting learning rate. If adagrad/adadelta/adam is
                    used, then this is the global learning rate. Recommended
                    settings: sgd = 1, adagrad = 0.1,
                    adadelta = 1, adam = 0.001""")
parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                    help="""If update_learning_rate, decay learning rate by
                    this much if (i) perplexity does not decrease on the
                    validation set or (ii) epoch has gone past
                    start_decay_at""")
parser.add_argument('-start_decay_at', type=int, default=8,
                    help="""Start decaying every epoch after and including this
                    epoch""")

# pretrained word vectors

parser.add_argument('-pre_word_vecs_enc',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the encoder side.
                    See README for specific formatting instructions.""")
parser.add_argument('-pre_word_vecs_dec',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the decoder side.
                    See README for specific formatting instructions.""")
parser.add_argument('-pre_word_vecs',
                    help="""If a valid path is specified, then this will load
                        pretrained word embeddings on the language model.
                        See README for specific formatting instructions.""")

# GPU
parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")

parser.add_argument('-log_interval', type=int, default=50,
                    help="Print stats at this interval.")

parser.add_argument('-seed', type=int, default=-1,
                    help="""Random seed used for the experiments
                    reproducibility.""")

opt = parser.parse_args()

print(opt)

if opt.seed > 0:
    torch.manual_seed(opt.seed)

if torch.cuda.is_available() and not opt.gpus:
    print("WARNING: You have a CUDA device, should run with -gpus 0")

if opt.gpus:
    cuda.set_device(opt.gpus[0])
    if opt.seed > 0:
        torch.cuda.manual_seed(opt.seed)


def NMTCriterion(vocabSize):
    weight = torch.ones(vocabSize)
    weight[onmt.Constants.PAD] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if opt.gpus:
        crit.cuda()
    return crit


def memoryEfficientLoss(outputs, targets, generator, crit, eval=False):
    # compute generations one piece at a time
    num_correct, loss = 0, 0
    outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval)

    batch_size = outputs.size(1)
    outputs_split = torch.split(outputs, opt.max_generator_batches)
    targets_split = torch.split(targets, opt.max_generator_batches)
    for i, (out_t, targ_t) in enumerate(zip(outputs_split, targets_split)):
        out_t = out_t.view(-1, out_t.size(2))
        scores_t = generator(out_t)
        loss_t = crit(scores_t, targ_t.view(-1))
        pred_t = scores_t.max(1)[1]
        num_correct_t = pred_t.data.eq(targ_t.data) \
                                   .masked_select(
                                       targ_t.ne(onmt.Constants.PAD).data) \
                                   .sum()
        num_correct += num_correct_t
        loss += loss_t.data[0]
        if not eval:
            loss_t.div(batch_size).backward()

    grad_output = None if outputs.grad is None else outputs.grad.data
    return loss, grad_output, num_correct


def eval(model, criterion, data):
    total_loss = 0
    total_words = 0
    total_num_correct = 0

    model.eval()
    for i in range(len(data)):
        # exclude original indices
        batch = data[i][:-1]
        outputs = model(batch)
        # exclude <s> from targets
        targets = batch[1][1:]
        loss, _, num_correct = memoryEfficientLoss(
                outputs, targets, model.generator, criterion, eval=True)
        total_loss += loss
        total_num_correct += num_correct
        total_words += targets.data.ne(onmt.Constants.PAD).sum()

    model.train()
    return total_loss / total_words, total_num_correct / total_words


def trainModel(model, trainData, validData, dataset, optim, opt):
    print(model)
    model.train()

    # Define criterion of each GPU.
    criterion = NMTCriterion(dataset['dicts']['tgt'].size())

    start_time = time.time()

    def trainEpoch(epoch, iter):

        if opt.extra_shuffle and epoch > opt.curriculum:
            trainData.shuffle()

        # Shuffle mini batch order.
        batchOrder = torch.randperm(len(trainData))

        total_loss, total_words, total_num_correct = 0, 0, 0
        report_loss, report_tgt_words = 0, 0
        report_src_words, report_num_correct = 0, 0
        start = time.time()
        for i in range(len(trainData)):

            if iter >= opt.epochs:
                break
            iter += 1

            batchIdx = batchOrder[i] if epoch > opt.curriculum else i
            # Exclude original indices.
            batch = trainData[batchIdx][:-1]

            model.zero_grad()
            outputs = model(batch)
            # Exclude <s> from targets.
            targets = batch[1][1:]
            loss, gradOutput, num_correct = memoryEfficientLoss(
                    outputs, targets, model.generator, criterion)

            outputs.backward(gradOutput)

            # Update the parameters.
            optim.step()

            num_words = targets.data.ne(onmt.Constants.PAD).sum()
            report_loss += loss
            report_num_correct += num_correct
            report_tgt_words += num_words
            report_src_words += batch[0][1].data.sum()
            total_loss += loss
            total_num_correct += num_correct
            total_words += num_words
            if i % opt.log_interval == -1 % opt.log_interval:
                print(("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; " +
                       "%3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed") %
                      (epoch, i+1, len(trainData),
                       report_num_correct / report_tgt_words * 100,
                       math.exp(report_loss / report_tgt_words),
                       report_src_words/(time.time()-start),
                       report_tgt_words/(time.time()-start),
                       time.time()-start_time))

                report_loss, report_tgt_words = 0, 0
                report_src_words, report_num_correct = 0, 0
                start = time.time()

            if iter % opt.save_each == 0:
                #  (2) evaluate on the validation set
                valid_loss, valid_acc = eval(model, criterion, validData)
                valid_ppl = math.exp(min(valid_loss, 100))
                print('Validation perplexity: %g' % valid_ppl)
                print('Validation accuracy: %g' % (valid_acc * 100))

                #  (3) update the learning rate
                if opt.use_learning_rate_decay:
                    optim.updateLearningRate(valid_ppl, iter)

                model_state_dict = (model.module.state_dict() if len(opt.gpus) > 1
                                    else model.state_dict())
                model_state_dict = {k: v for k, v in model_state_dict.items()
                                    if 'generator' not in k}
                generator_state_dict = (model.generator.module.state_dict()
                                        if len(opt.gpus) > 1
                                        else model.generator.state_dict())
                #  (4) drop a checkpoint
                checkpoint = {
                    'model': model_state_dict,
                    'generator': generator_state_dict,
                    'dicts': dataset['dicts'],
                    'opt': opt,
                    'epoch': epoch,
                    'optim': optim,
                    'type': opt.model_type
                }
                torch.save(checkpoint,
                           '%s_acc_%.2f_ppl_%.2f_iter%d_e%d.pt'
                           % (opt.save_model, 100 * valid_acc, valid_ppl, iter, epoch))

        return total_loss / total_words, total_num_correct / total_words, iter

    epoch, iter = 1, 0
    while iter < opt.epochs:
        print('')
        #  (1) train for one epoch on the training set
        train_loss, train_acc, iter = trainEpoch(epoch, iter)
        epoch += 1
        train_ppl = math.exp(min(train_loss, 100))
        print('Train perplexity: %g' % train_ppl)
        print('Train accuracy: %g' % (train_acc*100))


def main():
    print("Loading data from '%s'" % opt.data)

    dataset = torch.load(opt.data)
    if opt.model_type == 'nmt':
        if dataset.get("type", "text") not in ["bitext", "text"]:
            print("WARNING: The provided dataset is not bilingual!")
    elif opt.model_type == 'lm':
        if dataset.get("type", "text") != 'monotext':
            print("WARNING: The provided dataset is not monolingual!")
    else:
        raise NotImplementedError('Not valid model type %s' % opt.model_type)

    dict_checkpoint = (opt.train_from if opt.train_from
                       else opt.train_from_state_dict)
    if dict_checkpoint:
        print('Loading dicts from checkpoint at %s' % dict_checkpoint)
        checkpoint = torch.load(dict_checkpoint)
        if opt.model_type == 'nmt':
            assert checkpoint.get('type', None) is None or \
                checkpoint['type'] == "nmt", \
                "The loaded model is not neural machine translation!"
        elif opt.model_type == 'lm':
            assert checkpoint['type'] == "lm", \
                "The loaded model is not a language model!"
        dataset['dicts'] = checkpoint['dicts']

    trainData = onmt.Dataset(dataset['train']['src'],
                             dataset['train']['tgt'], opt.batch_size, opt.gpus,
                             data_type=dataset.get("type", "text"))
    validData = onmt.Dataset(dataset['valid']['src'],
                             dataset['valid']['tgt'], opt.batch_size, opt.gpus,
                             volatile=True,
                             data_type=dataset.get("type", "text"))

    dicts = dataset['dicts']
    model_opt = checkpoint['opt'] if dict_checkpoint else opt
    if dicts.get('tgt', None) is None:
        # Makes the code compatible with the language model
        dicts['tgt'] = dicts['src']
    if opt.model_type == 'nmt':
        print(' * vocabulary size. source = %d; target = %d' %
              (dicts['src'].size(), dicts['tgt'].size()))
    elif opt.model_type == 'lm':
        print(' * vocabulary size = %d' %
              (dicts['src'].size()))
    print(' * number of training sentences. %d' %
          len(dataset['train']['src']))
    print(' * maximum batch size. %d' % opt.batch_size)

    print('Building model...')

    if opt.model_type == 'nmt':

        decoder = onmt.Decoders.getDecoder(model_opt.decoder_type)(model_opt, dicts['tgt'])
        encoder = onmt.Encoders.getEncoder(model_opt.encoder_type)(model_opt, dicts['src'])

        model = onmt.Models.NMTModel(encoder, decoder)

    elif opt.model_type == 'lm':
        model = onmt.LanguageModel.LM(model_opt, dicts['src'])

    generator = nn.Sequential(
        nn.Linear(model_opt.rnn_size, dicts['tgt'].size()),
        nn.LogSoftmax())

    if opt.train_from:
        print('Loading model from checkpoint at %s' % opt.train_from)
        chk_model = checkpoint['model']
        generator_state_dict = chk_model.generator.state_dict()
        model_state_dict = {k: v for k, v in chk_model.state_dict().items()
                            if 'generator' not in k}
        model.load_state_dict(model_state_dict)
        generator.load_state_dict(generator_state_dict)
        opt.start_epoch = checkpoint['epoch'] + 1

    if opt.train_from_state_dict:
        print('Loading model from state_dict at %s'
              % opt.train_from_state_dict)
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
        model_opt.start_epoch = opt.start_epoch
        model_opt.epochs = opt.epochs

    if len(opt.gpus) >= 1:
        model.cuda()
        generator.cuda()
    else:
        model.cpu()
        generator.cpu()

    if len(opt.gpus) > 1:
        model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)
        generator = nn.DataParallel(generator, device_ids=opt.gpus, dim=0)
        model_opt["gpus"] = opt.gpus

    model.generator = generator

    if not opt.train_from_state_dict and not opt.train_from:
        for p in model.parameters():
            #p.data.uniform_(-opt.param_init, opt.param_init)
            if len(p.data.size()) > 1:
                init.xavier_normal(p.data)
            else:
                p.data.uniform_(-opt.param_init, opt.param_init)
        model.initialize_parameters(opt.param_init)
        model.load_pretrained_vectors(opt)

    if (not opt.train_from_state_dict and not opt.train_from) or opt.change_optimizer:
        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at
        )
        optim.set_parameters(model.parameters())
        model_opt.learning_rate = opt.learning_rate
        model_opt.learning_rate_decay = opt.learning_rate_decay
        model_opt.save_each = opt.save_each

    else:
        print('Loading optimizer from checkpoint:')
        optim = checkpoint['optim']
        optim.optimizer.load_state_dict(
            checkpoint['optim'].optimizer.state_dict())
        optim.set_parameters(model.parameters())

    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)

    if opt.train_from or opt.train_from_state_dict:
        print(model_opt)

    model_opt.use_learning_rate_decay = opt.use_learning_rate_decay
    trainModel(model, trainData, validData, dataset, optim, model_opt)


if __name__ == "__main__":
    main()
