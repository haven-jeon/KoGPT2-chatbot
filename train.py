# -*- coding: utf-8 -*-
import argparse
import logging
import math

import gluonnlp as nlp
import mxnet as mx
import pandas as pd
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.mxnet_kogpt2 import get_mxnet_kogpt2_model
from kogpt2.utils import get_tokenizer
from mxnet import gluon, nd
from mxnet.gluon import nn

parser = argparse.ArgumentParser(description='Simsimi based on KoGPT-2')

parser.add_argument('--num-epoch',
                    type=int,
                    default=1,
                    help='number of iterations to train (default: 2)')

parser.add_argument('--max-seq-len',
                    type=int,
                    default=32,
                    help='max sentence length on input (default: 32)')

parser.add_argument('--batch-size',
                    type=int,
                    default=64,
                    help='batch size for training (default: 64)')

parser.add_argument('--chat',
                    action='store_true',
                    default=False,
                    help='response generation on given user input')

parser.add_argument('--sentiment',
                    type=str,
                    default='0',
                    help='sentiment for system. 0 is neutral, 1 is negative, 2 is positive.')


parser.add_argument('--model_params',
                    type=str,
                    default='kogpt2_chat.params',
                    help='model binary for starting chat')

parser.add_argument('--train',
                    action='store_true',
                    default=False,
                    help='eval train set (default: False)')


parser.add_argument(
    '--accumulate',
    type=int,
    default=1,
    help='accumulate gradient to achieve the same result with a large batch size')

opt = parser.parse_args()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '<s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'


class ChatDataset(gluon.data.Dataset):
    def __init__(self, chats, tok_path, vocab, max_len=32):
        self._data = chats
        self._tok_path = tok_path
        self.tokenizer = None
        self.first = True
        self.q_token = U_TKN
        self.a_token = S_TKN
        self.sent_token = SENT
        self.bos = BOS
        self.eos = EOS
        self.maskt = MASK
        self.vocab = vocab
        self.max_len = max_len
        self.padder = nlp.data.PadSequence(
            max_len, pad_val=self.vocab[self.vocab.padding_token])

    def _activate_sp(self):
        self.tokenizer = nlp.data.SentencepieceTokenizer(self._tok_path, 0, 0)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        if self.tokenizer is None:
            self._activate_sp()
        turn = self._data.iloc[idx]
        q = turn['Q']
        a = turn['A']
        sentiment = str(turn['label'])
        q_toked = [
            self.q_token,
        ] + self.tokenizer(q) + [
            self.eos,
        ] + [self.sent_token] + self.tokenizer(sentiment) + [
            self.eos,
        ]
        q_len = len(q_toked)
        a_toked = [
            self.a_token,
        ] + self.tokenizer(a) + [
            self.eos,
        ]
        a_len = len(a_toked)
        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len
            if a_len <= 0:
                q_toked = q_toked[-(int(self.max_len/2)):]
                q_len = len(q_toked)
                a_len = self.max_len - q_len
                assert a_len > 0
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)
            assert a_len == len(a_toked), f'{a_len} ==? {len(a_toked)}'
        # [<mask>, <mask>, ...., <mask>, ..., A.. <eos>, <pad>....]
        labels = [
            self.maskt,
        ] * q_len + a_toked[1:]
        if self.first:
            logging.info("contexts : {}".format(q))
            logging.info("toked ctx: {}".format(q_toked))
            logging.info("response : {}".format(a))
            logging.info("toked response : {}".format(a_toked))
            logging.info('labels {}'.format(labels))
            self.first = False
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        return (self.padder(self.vocab[q_toked + a_toked]), nd.array(mask),
                self.padder(self.vocab[labels]))


class KoGPT2Chat(nn.HybridBlock):
    def __init__(self, kogpt2, prefix=None, params=None):
        super(KoGPT2Chat, self).__init__(prefix=prefix, params=params)
        self.kogpt2 = kogpt2

    def hybrid_forward(self, F, inputs):
        # (batch, seq_len, hiddens)
        output, _ = self.kogpt2(inputs)
        return output


if mx.context.num_gpus() > 0:
    ctx = mx.gpu()
else:
    ctx = mx.cpu()


def train():
    tok_path = get_tokenizer()
    model, vocab = get_mxnet_kogpt2_model(ctx=ctx)
    # tok = SentencepieceTokenizer(tok_path, num_best=0, alpha=0)

    data = pd.read_csv('Chatbot_data/ChatbotData.csv')

    max_len = opt.max_seq_len
    train_set = ChatDataset(data, tok_path, vocab, max_len=max_len)
    batch_size = opt.batch_size

    train_dataloader = mx.gluon.data.DataLoader(train_set,
                                                batch_size=batch_size,
                                                num_workers=5,
                                                shuffle=True)
    kogptqa = KoGPT2Chat(model)
    kogptqa.hybridize()

    # softmax cross entropy loss for classification
    loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
    loss_function.hybridize()

    num_epochs = opt.num_epoch
    lr = 5e-5
    trainer = gluon.Trainer(kogptqa.collect_params(), 'bertadam', {
        'learning_rate': lr,
        'epsilon': 1e-8,
        'wd': 0.01
    })
    # LayerNorm과 Bias에는 Weight Decay를 적용하지 않는다.
    for _, v in kogptqa.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    params = [
        p for p in kogptqa.collect_params().values() if p.grad_req != 'null'
    ]
    # learning rate warmup
    accumulate = opt.accumulate
    step_size = batch_size * accumulate if accumulate else batch_size
    num_train_examples = len(train_set)
    num_train_steps = int(num_train_examples / step_size * num_epochs)
    warmup_ratio = 0.1
    num_warmup_steps = int(num_train_steps * warmup_ratio)
    step_num = 0
    all_model_params = kogptqa.collect_params()

    log_interval = 50
    neg = -1e18
    # Set grad_req if gradient accumulation is required
    if accumulate and accumulate > 1:
        for p in params:
            p.grad_req = 'add'

    for epoch_id in range(num_epochs):
        step_loss = 0
        for batch_id, (token_ids, mask, label) in enumerate(train_dataloader):
            if step_num < num_warmup_steps:
                new_lr = lr * step_num / num_warmup_steps
            else:
                non_warmup_steps = step_num - num_warmup_steps
                offset = non_warmup_steps / (num_train_steps -
                                             num_warmup_steps)
                new_lr = lr - offset * lr
            trainer.set_learning_rate(new_lr)
            with mx.autograd.record():
                # load data to GPU or GPU
                token_ids = token_ids.as_in_context(ctx)
                mask = mask.as_in_context(ctx)
                label = label.as_in_context(ctx)
                # forward computation
                out = kogptqa(token_ids)
                masked_out = nd.where(
                    mask.expand_dims(axis=2).repeat(repeats=out.shape[2],
                                                    axis=2), out,
                    neg * nd.ones_like(out))
                # loss for responses exincluding MASK and PAD
                ls = loss_function(masked_out, label).sum() / mask.sum()
            # backward computation
            ls.backward()
            if not accumulate or (batch_id + 1) % accumulate == 0:
                trainer.allreduce_grads()
                nlp.utils.clip_grad_global_norm(params, 1)
                trainer.update(accumulate if accumulate else 1)
                step_num += 1
                if accumulate and accumulate > 1:
                    # set grad to zero for gradient accumulation
                    all_model_params.zero_grad()
            step_loss += ls.asscalar()
            if step_num % log_interval == 0 and step_num > 0:
                print(
                    '[Epoch {} Batch {}/{}] loss={:.4f}, lr={:.10f}, train ppl={:.3f}'
                    .format(epoch_id + 1, batch_id + 1, len(train_dataloader),
                            step_loss / log_interval, trainer.learning_rate,
                            math.exp(step_loss / log_interval)))
                step_loss = 0
    logging.info('saving model file to {}'.format(opt.model_params))
    kogptqa.save_parameters(opt.model_params)


def chat(model_params, sent='0'):
    tok_path = get_tokenizer()
    model, vocab = get_mxnet_kogpt2_model(ctx=ctx)
    tok = SentencepieceTokenizer(tok_path, num_best=0, alpha=0)
    kogptqa = KoGPT2Chat(model)
    kogptqa.load_parameters(model_params, ctx=ctx)
    sent_tokens = tok(sent)
    while 1:
        q = input('user > ').strip()
        if q == 'quit':
            break
        q_tok = tok(q)
        a = ''
        a_tok = []
        while 1:
            input_ids = mx.nd.array([vocab[U_TKN]] + vocab[q_tok] +
                                    vocab[EOS, SENT] + vocab[sent_tokens] +
                                    vocab[EOS, S_TKN] +
                                    vocab[a_tok]).expand_dims(axis=0)
            pred = kogptqa(input_ids.as_in_context(ctx))
            gen = vocab.to_tokens(
                mx.nd.argmax(
                    pred,
                    axis=-1).squeeze().astype('int').asnumpy().tolist())[-1]
            if gen == EOS:
                break
            a += gen.replace('▁', ' ')
            a_tok = tok(a)
        print("Simsimi > {}".format(a.strip()))


if __name__ == "__main__":
    if opt.train:
        train()
    if opt.chat:
        chat(opt.model_params, opt.sentiment)
