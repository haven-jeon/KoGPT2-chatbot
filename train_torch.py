import argparse
import logging

import gluonnlp as nlp
import numpy as np
import pandas as pd
import torch
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from kogpt2.utils import get_tokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW

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


class CharDataset(Dataset):
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
            remains = self.max_len - q_len
            a_len = remains
            a_toked = a_toked[-a_len:]
            assert a_len == len(a_toked)
        # [mask, mask, ...., mask, ..., <bos>,..A.. <eos>, <pad>....]
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
        return (self.padder(self.vocab[q_toked + a_toked]), np.array(mask),
                self.padder(self.vocab[labels]))


class KoGPT2Chat(LightningModule):
    def __init__(self, max_len=32, batch_size=64, lr=5e-5, num_epochs=1):
        super(KoGPT2Chat, self).__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.max_len = max_len
        self.tok_path = get_tokenizer()
        self.num_epochs = num_epochs
        self.neg = -1e18
        self.kogpt2, self.vocab = get_pytorch_kogpt2_model()
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs):
        # (batch, seq_len, hiddens)
        output, _ = self.kogpt2(inputs)
        return output

    def training_step(self, batch, batch_idx):
        token_ids, mask, label = batch
        out = self(token_ids)
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
        loss = self.loss_function(mask_out.transpose(2, 1), label)
        loss_avg = loss.sum() / mask.sum()
        tensorboard_logs = {'train_loss': loss_avg}
        return {'loss': loss_avg, 'log': tensorboard_logs}

    # learning rate warm-up
    def optimizer_step(self, current_epoch, batch_idx, optimizer,
                       optimizer_idx, second_order_closure=None):
        # warm up lr
        num_train_steps = int(len(self.train_set) / self.batch_size * self.num_epochs)
        warmup_ratio = 0.1
        num_warmup_steps = int(num_train_steps * warmup_ratio)
        # update params
        step_num = self.trainer.global_step
        for pg in optimizer.param_groups:
            if step_num < num_warmup_steps:
                new_lr = self.lr * step_num / num_warmup_steps
            else:
                non_warmup_steps = step_num - num_warmup_steps
                offset = non_warmup_steps / (num_train_steps - num_warmup_steps)
                new_lr = self.lr - offset * self.lr
            pg['lr'] = new_lr
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.lr, correct_bias=False)
        return optimizer

    def _collate_fn(self, batch):
        data = [item[0] for item in batch]
        mask = [item[1] for item in batch]
        label = [item[2] for item in batch]
        return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

    def train_dataloader(self):
        data = pd.read_csv('Chatbot_data/ChatbotData.csv')
        self.train_set = CharDataset(data, self.tok_path, self.vocab, max_len=self.max_len)
        train_dataloader = DataLoader(
            self.train_set, batch_size=self.batch_size, num_workers=2,
            shuffle=True, collate_fn=self._collate_fn)
        return train_dataloader


def chat(kogptqa, sent='0'):
    tok_path = get_tokenizer()
    _, vocab = get_pytorch_kogpt2_model()
    tok = SentencepieceTokenizer(tok_path, num_best=0, alpha=0)
    sent_tokens = tok(sent)
    with torch.no_grad():
        while 1:
            q = input('user > ').strip()
            if q == 'quit':
                break
            q_tok = tok(q)
            a = ''
            a_tok = []
            while 1:
                input_ids = torch.LongTensor([
                    vocab[U_TKN]] + vocab[q_tok] +
                    vocab[EOS, SENT] + vocab[sent_tokens] +
                    vocab[EOS, S_TKN] +
                    vocab[a_tok]).unsqueeze(dim=0)
                pred = kogptqa(input_ids)
                gen = vocab.to_tokens(
                    torch.argmax(
                        pred,
                        dim=-1).squeeze().numpy().tolist())[-1]
                if gen == EOS:
                    break
                a += gen.replace('▁', ' ')
                a_tok = tok(a)
            print("Simsimi > {}".format(a.strip()))


if __name__ == "__main__":
    if opt.train:
        checkpoint_callback = ModelCheckpoint(
            filepath='model_chp/{epoch:02d}-{loss:.2f}',
            verbose=True,
            save_last=True,
            monitor='loss',
            mode='min',
            prefix='model_'
        )
        model = KoGPT2Chat(max_len=opt.max_seq_len, batch_size=opt.batch_size, num_epochs=opt.num_epoch)
        model.train()
        trainer = Trainer(
            gpus=1, max_epochs=opt.num_epoch,
            checkpoint_callback=checkpoint_callback)
        trainer.fit(model)
        logging.info('best model path {}'.format(checkpoint_callback.best_model_path))
    if opt.chat:
        model = KoGPT2Chat.load_from_checkpoint('model_chp/model_last.ckpt')
        model.freeze()
        chat(model)
