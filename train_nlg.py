import argparse
import time
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from joblib import Parallel, delayed
from sklearn import metrics
from tez import Tez, TezConfig
from tez.callbacks import EarlyStopping
from tez.utils import seed_everything, AverageMeter
from tez import enums
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_polynomial_decay_schedule_with_warmup
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Model, GPT2PreTrainedModel
from transformers import BartForConditionalGeneration, BartForCausalLM, BartTokenizerFast, BartModel, BartPretrainedModel
import bitsandbytes as bnb

import difflib

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=False, default='test')
    parser.add_argument("--lr", type=float, required=False, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100, required=False)
    parser.add_argument("--save_idx", type=int, default=0, required=False)

    #parser.add_argument("--output", type=str, default="model", required=False)
    parser.add_argument("--max_len", type=int, default=1024, required=False)
    parser.add_argument("--batch_size", type=int, default=4, required=False)
    parser.add_argument("--valid_batch_size", type=int, default=8, required=False)
    parser.add_argument("--freeze", type=int, default=2, required=False)

    parser.add_argument("--accumulation_steps", type=int, default=1, required=False)

    parser.add_argument("--load", action="store_true", required=False)
    parser.add_argument("--fp16", action="store_true", required=False)
    parser.add_argument("--predict", action="store_true", required=False)
    return parser.parse_args()

args = parse_args()

torch.multiprocessing.set_sharing_strategy("file_system")

warnings.filterwarnings("ignore")

def get_dict(loss, avg_prob):
    return_dict = {
		"loss" : torch.tensor(loss),
		"avg_prob": torch.tensor(avg_prob),
            }
    return return_dict

def _prepare_data_helper(tok, df, text_ids): # ko_nlg
    # tok - totkenizer
    # df - dataframe containes query, answer_0, answer_1, preference, and text_ids
    # preference 0 means answer_0 is bettter, and 1 means answer_1 is bettter
    samples = []
    for idx in tqdm(df['text_id'].values):
        tmp_df = df[df.text_id==idx].reset_index(drop=True)
        Q   = tmp_df.Q.values[0]
        A   = tmp_df.A.values[0]

        enc_Q   = tok.encode_plus(f'{Q}',
            add_special_tokens=False)#, max_length=args.max_len*100-2)
        enc_A   = tok.encode_plus(f'{A}',
            add_special_tokens=False)#, max_length=args.max_len-2)

        sample = {
            "Q": enc_Q['input_ids'][-args.max_len-2:],
            "A": enc_A['input_ids'][-args.max_len-2:],
            "text_id": idx,
        }
        samples.append(sample)
    return samples

def prepare_data(df, tok, num_jobs): # ko_nlg
    samples = []
    text_ids = df["text_id"].unique()

    text_ids_splits = np.array_split(text_ids, num_jobs)

    results = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
        delayed(_prepare_data_helper)(tok, df, idx) for idx in text_ids_splits
    )
    for result in results:
        samples.extend(result)

    return samples


class PreferenceDataset:
    def __init__(self, samples, max_len, tokenizer):
        self.samples = samples
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        Q   = self.samples[idx]["Q"]
        A   = self.samples[idx]["A"]

        enc_ids = [0] + Q + [2]
        dec_ids = [0] + A + [2]
        enc_mask = [1]*len(enc_ids)
        dec_mask = [1]*len(dec_ids)

        res = {
            "enc_ids": enc_ids,
            "dec_ids": dec_ids,
            "enc_mask": enc_mask,
            "dec_mask": dec_mask,
       }
        return res


class Collate:
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        output = dict()
        output["enc_ids" ] = [sample["enc_ids" ] for sample in batch]
        output["enc_mask"] = [sample["enc_mask"] for sample in batch]
        output["dec_ids" ] = [sample["dec_ids" ] for sample in batch]
        output["dec_mask"] = [sample["dec_mask"] for sample in batch]

        enc_max = min(max([len(ids) for ids in output["enc_ids"]]), self.max_len)
        dec_max = min(max([len(ids) for ids in output["dec_ids"]]), self.max_len)

        output["enc_ids" ] = [s + (enc_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["enc_ids"]]
        output["enc_mask"] = [s + (enc_max - len(s)) * [0] for s in output["enc_mask"]]
        output["dec_ids" ] = [s + (dec_max - len(s)) * [self.tokenizer.pad_token_id] for s in output["dec_ids"]]
        output["dec_mask"] = [s + (dec_max - len(s)) * [0] for s in output["dec_mask"]]

        # convert to tensors
        output["enc_ids" ] = torch.tensor(output["enc_ids" ], dtype=torch.long)
        output["enc_mask"] = torch.tensor(output["enc_mask"], dtype=torch.long)
        output["dec_ids" ] = torch.tensor(output["dec_ids" ], dtype=torch.long)
        output["dec_mask"] = torch.tensor(output["dec_mask"], dtype=torch.long)
        return output

class CustomModel(nn.Module):
    def __init__(self, num_train_steps, learning_rate, steps_per_epoch):
        super().__init__()
        self.num_train_steps = num_train_steps
        self.learning_rate = learning_rate
        self.steps_per_epoch = steps_per_epoch

        # config_rew is the same as the config_rew
        self.model_name = 'hyunwoongko/kobart'

        # policy and target policy networks
        self.output_net = BartForConditionalGeneration.from_pretrained(self.model_name)
        #self.freeze(self.output_net.model.decoder.embed_tokens)
        self.loss_fct = CrossEntropyLoss(reduce=False)

    def freeze(self, module):
        for p in module.parameters():
            p.requires_grad = False
            p.grad = None

    def optimizer_scheduler(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.001,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        opt = torch.optim.AdamW(optimizer_parameters, lr=self.learning_rate)
        if True:
            opt = bnb.optim.AdamW(optimizer_parameters, lr=self.learning_rate, optim_bits=8)
            #self.set_embedding_parameters_bits(embeddings_path=self.transformer.embeddings)

        sch = get_polynomial_decay_schedule_with_warmup(
            opt,
            num_warmup_steps=int(self.num_train_steps * 0.2),
            num_training_steps=self.num_train_steps,
            last_epoch=-1,
        )
        return opt, sch


    def forward(self, enc_ids, dec_ids=None, enc_mask=None, dec_mask=None, target=None):
    #def forward(self, output_ids=None, output_mask=None, loss_mask=None, len_output=None):
        out = self.output_net(input_ids=enc_ids, attention_mask=enc_mask,
                              decoder_input_ids=dec_ids[:, :-1], decoder_attention_mask=dec_mask[:, :-1],)
                              #labels=dec_ids[:, 1:])

        logit_loss = 0
        avg_prob = 0
        ## generation loss
        for logits, ids, mask in zip(out['logits'], dec_ids, dec_mask):
            len_ = mask.sum()
            shift_ids = ids[1:len_]
            shift_logits = logits[:len_-1]
            loss = (self.loss_fct(shift_logits, shift_ids)).sum()
            prob = torch.softmax(shift_logits, 1).gather(1, shift_ids.unsqueeze(0).detach())
            avg_prob = avg_prob + prob.sum()
            logit_loss = logit_loss + loss

        logit_loss = logit_loss / dec_mask.sum()
        avg_prob = avg_prob / dec_mask.sum()
        loss_ = logit_loss
        return 0, loss_, get_dict(loss_, avg_prob)

NUM_JOBS = 4
seed_everything(42)

train_df = pd.read_csv('data_nlg/train_simple.csv')#.head(500000)
valid_df = pd.read_csv('data_nlg/valid_simple.csv').head(50000)
train_df['text_id'] = np.arange(len(train_df))
valid_df['text_id'] = np.arange(len(valid_df))

tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join('tokenizer/emji_tokenizer/model.json'),
            bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')

tok = tokenizer

df = train_df.copy()

train_samples = prepare_data(train_df, tokenizer,  num_jobs=NUM_JOBS)
valid_samples = prepare_data(valid_df, tokenizer,  num_jobs=NUM_JOBS)
#valid_samples = list(sorted(valid_samples, key=lambda d: len(d[""])))

train_dataset = PreferenceDataset(train_samples, args.max_len, tokenizer)
valid_dataset = PreferenceDataset(valid_samples, args.max_len, tokenizer)

num_train_steps = int(len(train_dataset) / args.batch_size / args.accumulation_steps * args.epochs)
n_gpu = torch.cuda.device_count()
num_train_steps /= n_gpu

collate_fn = Collate(tokenizer, args.max_len)

model = CustomModel(
    num_train_steps=num_train_steps,
    learning_rate=args.lr,
    steps_per_epoch=len(train_dataset) / args.batch_size,
)

save_path = 'model_nlg.bin'
model = Tez(model)
es = EarlyStopping(
    monitor="valid_avg_prob",
    model_path=os.path.join(save_path),
    patience=5,
    mode="max",
    delta=0.0001,
    save_weights_only=True,
)

config = TezConfig(
    training_batch_size=args.batch_size,
    validation_batch_size=args.valid_batch_size,
    #validation_batch_size=args.valid_batch_size,
    gradient_accumulation_steps=args.accumulation_steps,
    epochs=args.epochs,
    #fp16=False,
    fp16=args.fp16,
    valid_shuffle=False,
    step_scheduler_after="batch",
    num_jobs=4,
    val_strategy="batch",
    val_steps=10000,
    #val_steps=args.val_steps,
)
if args.load:
    model.model.load_state_dict(torch.load('./model_chat_simple.bin'));

model.fit(
    train_dataset, valid_dataset=valid_dataset,
    train_collate_fn=collate_fn, valid_collate_fn=collate_fn,
    config=config, callbacks=[es],
)