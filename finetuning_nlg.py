import argparse
import time
import os
import warnings
import gc

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.utils.checkpoint
from torch.utils.data import DataLoader
from joblib import Parallel, delayed
from sklearn import metrics
from tez import Tez, TezConfig
from tez.callbacks import EarlyStopping
from tez.utils import seed_everything, AverageMeter
from tez import enums
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_polynomial_decay_schedule_with_warmup
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Model, GPT2PreTrainedModel
from transformers import BartForCausalLM, BartTokenizerFast, BartModel, BartPretrainedModel, BartForConditionalGeneration
from transformers import PreTrainedTokenizerFast
import bitsandbytes as bnb

os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.multiprocessing.set_sharing_strategy("file_system")

warnings.filterwarnings("ignore")


avg_score = []
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=False, default='test')
    parser.add_argument("--lr", type=float, required=False, default=1e-6)
    parser.add_argument("--tau", type=float, required=False, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10, required=False)
    parser.add_argument("--max_len", type=int, default=128, required=False)
    parser.add_argument("--batch_size", type=int, default=4, required=False)
    parser.add_argument("--freeze", type=int, default=2, required=False)
    parser.add_argument("--save_period", type=int, default=10000, required=False)

    parser.add_argument("--ppo_k", type=int, default=1, required=False)
    parser.add_argument("--kl_avg", type=float, default=2.0)

    parser.add_argument("--kl_div", type=bool, default=True)
    parser.add_argument("--mean_reward", type=bool, default=False)
    parser.add_argument("--R_as_adv", type=bool, default=False)

    parser.add_argument("--accumulation_steps", type=int, default=1, required=False)

    return parser.parse_args()

def get_dict(loss, ethics, std):
    return_dict = {"loss": torch.tensor(loss),
            "ethics": torch.tensor(ethics),
            "ethics_std": torch.tensor(std),
            }
    return return_dict


def _prepare_data_helper(tok, df, text_ids): # ko_nlg
    # tok - totkenizer
    # df - dataframe containes query, answer_0, answer_1, preference, and text_ids
    # preference 0 means answer_0 is bettter, and 1 means answer_1 is bettter
    samples = []
    for idx in tqdm(text_ids):
        tmp_df = df[df.text_id==idx].reset_index(drop=True)
        q = tmp_df['Q'].values[0]
        a = tmp_df['A'].values[0]
        enc_q = tok.encode_plus(q, add_special_tokens=False)
        enc_a = tok.encode_plus(a, add_special_tokens=False)

        input_ids_q, attention_mask_q = enc_q["input_ids"], enc_q["attention_mask"]
        input_ids_a, attention_mask_a = enc_a["input_ids"], enc_a["attention_mask"]
        sample = {
            "q": input_ids_q,
            "a": input_ids_a,
            "text_id": idx,
        }
        samples.append(sample)
    return samples


def prepare_data(df, tokenizer, num_jobs): # ko_nlg
    samples = []
    text_ids = df["text_id"].unique()

    text_ids_splits = np.array_split(text_ids, num_jobs)

    results = Parallel(n_jobs=num_jobs, backend="multiprocessing")(
        delayed(_prepare_data_helper)(tokenizer, df, idx) for idx in text_ids_splits
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
        q = self.samples[idx]["q"]
        a = self.samples[idx]["a"]

        q = [0]+q+[2]
        a = [1]+a+[2]

        m_q = [1]*len(q)
        m_a = [1]*len(a)

        res = {
            "q": q,
            "a": a,
            "m_q": m_q,
            "m_a": m_a,
        }
        return res


class Collate:
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        output = dict()
        output["q"] = [sample["q"] for sample in batch]
        output["a"] = [sample["a"] for sample in batch]
        output["m_q"] = [sample["m_q"] for sample in batch]
        output["m_a"] = [sample["m_a"] for sample in batch]

        # calculate max token length of this batch
        batch_max_q = max([len(q) for q in output["q"]])
        batch_max_a = max([len(a) for a in output["a"]])

        # add padding
        #if self.tokenizer.padding_side == "right":
        output["q"] = [s + (batch_max_q - len(s)) * [self.tokenizer.pad_token_id] for s in output["q"]]
        output["m_q"] = [s + (batch_max_q - len(s)) * [0] for s in output["m_q"]]
        output["a"] = [s + (batch_max_a - len(s)) * [self.tokenizer.pad_token_id] for s in output["a"]]
        output["m_a"] = [s + (batch_max_a - len(s)) * [0] for s in output["m_a"]]

        # convert to tensors
        output["q"]   = torch.tensor(output["q"]  , dtype=torch.long)
        output["m_q"] = torch.tensor(output["m_q"], dtype=torch.long)
        output["a"]   = torch.tensor(output["a"]  , dtype=torch.long)
        output["m_a"] = torch.tensor(output["m_a"], dtype=torch.long)
        return output

class CLSModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = 'hyunwoongko/kobart'
        self.output_net = AutoModel.from_pretrained(self.model_name)
        self.fin_net = nn.Linear(self.output_net.config.d_model, 1)

    def forward(self, enc_ids, enc_mask=None):
        out = self.output_net(input_ids=enc_ids, attention_mask=enc_mask,)
        fin_out = self.fin_net(out['encoder_last_hidden_state'])
        fin_out = torch.sigmoid(fin_out)
        return fin_out

class NLGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = 'hyunwoongko/kobart'
        self.output_net = BartForConditionalGeneration.from_pretrained(self.model_name)
        self.load_state_dict(torch.load("./model_nlg.bin"))


class CustomModel(nn.Module):
    def __init__(self, num_train_steps, learning_rate, steps_per_epoch):
        super().__init__()
        self.counter = 0
        self.num_train_steps = num_train_steps
        self.learning_rate = learning_rate
        self.steps_per_epoch = steps_per_epoch

        # config_rew is the same as the config_rew
        self.model_name = 'hyunwoongko/kobart'

        # preference networks (not reward but use r, following the article)
        self.r_net = CLSModel()
        self.r_net.load_state_dict(torch.load('./model_cls.bin'))

        # parameter for KL
        self.beta  = 1.
        self.s_net = BartForConditionalGeneration.from_pretrained(self.model_name)
        self.t_net = BartForConditionalGeneration.from_pretrained(self.model_name)
        self.s_out = nn.Linear(self.s_net.config.d_model, 1)
        self.t_out = nn.Linear(self.t_net.config.d_model, 1)

        tmp_model = NLGModel()
        self.s_net.load_state_dict(tmp_model.output_net.state_dict())
        del tmp_model
        gc.collect()

        self.update_target()
        self.freeze_init()

    def adapt_KL(self, current_kl, target=2 , clip=0.1):
        len_ = len(current_kl)
        kl_mean = torch.mean(current_kl).item()*self.beta
        cliped_error = np.clip(1-kl_mean/target, -clip, clip)
        multiplier = 1 + cliped_error*0.1*len_ #/ horizon e.g., horizon 10000
        self.beta *= multiplier
        return self.beta, kl_mean

    def update_target(self):
        self.t_net.load_state_dict(self.s_net.state_dict())
        self.t_out.load_state_dict(self.s_out.state_dict())

    def update_target_soft(self, tau=1e-3):
        for t_p, s_p in zip(self.t_net.parameters(), self.s_net.parameters()):
            t_p.data.copy_(tau*s_p.data+(1-tau)*t_p.data)
        for t_p, s_p in zip(self.t_out.parameters(), self.s_out.parameters()):
            t_p.data.copy_(tau*s_p.data+(1-tau)*t_p.data)

    def freeze_init(self):
        self.freeze(self.t_net)
        self.freeze(self.t_out)
        self.freeze(self.r_net)

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
        self.opt = bnb.optim.AdamW(optimizer_parameters, lr=self.learning_rate, optim_bits=8)
        #self.set_embedding_parameters_bits(embeddings_path=self.transformer.embeddings)

        sch = get_polynomial_decay_schedule_with_warmup(
            self.opt,
            num_warmup_steps=int(self.num_train_steps * 0.2),
            num_training_steps=self.num_train_steps,
            last_epoch=-1,
        )
        return self.opt, sch


    def gen_qa(self, q, m_q, a, m_a):
        a_gen = self.s_net.generate(q, max_length=64, do_sample=True, eos_token_id=2,
                                    num_beams=10)
        a_ids = torch.zeros(len(q)*2, max(a_gen.shape[-1], a.shape[-1]),
                            dtype=torch.int64).cuda()
        a_ids[:len(q), :a_gen.shape[-1]] = a_gen
        a_ids[len(q):, :a.shape[-1]] = a

        q_ids = torch.concat((q, q))
        q_masks = torch.concat((m_q, m_q))
        len_a_all = []
        len_qa_all = []
        for i, a in enumerate(a_ids):
            tmp = torch.where(a==2)
            if len(tmp[0])==0:
                len_a = len(a)
            else:
                len_a = (tmp[0][0]+1).item()
            len_a_all.append(len_a)
            len_qa_all.append(len_a+q_masks[i].sum().item())
        len_a_max = int(torch.Tensor(len_a_all).max().item())
        len_qa_max = int(torch.Tensor(len_qa_all).max().item())

        a_masks = torch.zeros((len(q_masks), len_a_max), dtype=torch.int64, device=q_masks.device)
        qa_masks = torch.zeros((len(q_masks), len_qa_max), dtype=torch.int64, device=q_masks.device)
        a_ids = a_ids[:, :len_a_max]
        qa_ids = torch.zeros((len(q_masks), len_qa_max), dtype=torch.int64, device=q_masks.device)

        for i, (q, a, q_mask, len_a) in enumerate(zip(q_ids, a_ids, q_masks, len_a_all)):
            len_q = q_mask.sum()
            a_masks[i, :len_a] = 1
            qa_masks[i, :len_q+len_a] = 1
            qa_ids[i, :len_q] = q[:len_q]
            qa_ids[i, len_q:len_q+len_a] = a[:len_a]
        return q_ids, q_masks, a_ids, a_masks, qa_ids, qa_masks

    def forward(self, q, m_q=None, a=None, m_a=None):
        self.r_net.eval(); self.s_net.eval(); self.t_net.eval()
        ########################
        q_ids, q_masks, a_ids, a_masks, qa_ids, qa_masks = self.gen_qa(q, m_q, a, m_a)

        score = 1-self.r_net(a_ids, a_masks).squeeze(-1)
        score = (score - 0.8)/0.20
        score_comp = torch.einsum('bc, bc-> b', score, a_masks*1.0)/a_masks.sum()
        #s_score, t_score = torch.mean(score[:len(q)]), torch.mean(score[len(q):])
        self.r_net.train(); self.s_net.train(); self.t_net.train()
        def prob_clip(prob, clip=1e-10):
            prob = torch.clip(prob, clip, 1-clip)
            return prob/prob.sum(dim=-1).unsqueeze(-1)
        for k in range(args.ppo_k):
            s_V = self.s_net(qa_ids, qa_masks)['encoder_last_hidden_state']
            s_V = self.s_out(s_V)
            s_pi = self.s_net(input_ids=q_ids, attention_mask=q_masks,
                    decoder_input_ids=a_ids, decoder_attention_mask=a_masks)['logits']
            with torch.no_grad():
                t_V = self.t_net(qa_ids, qa_masks)['encoder_last_hidden_state']
                t_V = self.t_out(t_V)
                t_pi = self.t_net(input_ids=q_ids, attention_mask=q_masks,
                    decoder_input_ids=a_ids, decoder_attention_mask=a_masks)['logits']
                if k == 0:
                    s_pi_old = self.s_net(input_ids=q_ids, attention_mask=q_masks,
                    decoder_input_ids=a_ids, decoder_attention_mask=a_masks)['logits'].softmax(dim=-1)
                    s_pi_old = prob_clip(s_pi)
            s_pi, t_pi = s_pi.softmax(dim=-1), t_pi.softmax(dim=-1)
            s_pi = prob_clip(s_pi)
            t_pi = prob_clip(t_pi)

            loss, score_mean, score_std = 0, 0, 0
            for i in range(len(q_ids)):
                l_q, l_a, l_qa = q_masks[i].sum(), a_masks[i].sum(), qa_masks[i].sum()
                v_  = s_V[i][l_q:l_qa-1]
                v_n = torch.zeros_like(v_)
                v_n[:-1] = t_V[i][l_q+1:l_qa-1]
                r_ = score[i][:l_a-1]
                score_mean = score_mean + (score[i][:l_a-1].mean())
                score_std = score_std + (score[i][:l_a-1].std())
                s_prob_g = torch.gather(s_pi[i], 1, a_ids[i].reshape(-1, 1))
                s_old_prob_g = torch.gather(s_pi_old[i], 1, a_ids[i].reshape(-1, 1))
                t_prob_g = torch.gather(t_pi[i], 1, a_ids[i].reshape(-1, 1))
                s_prob_g = prob_clip(s_prob_g)
                t_prob_g = prob_clip(t_prob_g)
                s_old_prob_g = prob_clip(s_old_prob_g)
                with torch.no_grad():
                    if args.kl_div:
                        kl_div = torch.log(s_pi[i][:l_a-1]) - torch.log(t_pi[i][:l_a-1])
                        kl_div = torch.einsum('bl,bl->b', s_pi[i][:l_a-1], kl_div)
                    else:
                        kl_div = torch.log(s_prob_g[:l_a-1]) - torch.log(t_prob_g[:l_a-1])
                if args.mean_reward:
                    R_ = r_.mean() - self.beta*kl_div
                else:
                    R_ = r_ - self.beta*kl_div
                beta, mean_kl = self.adapt_KL(kl_div, target=args.kl_avg)

                val_loss, pol_loss = 0, 0
                with torch.no_grad():
                    last_gae_lambda = 0
                    adv_ = []
                    for j in range(1, l_a):
                        # gamma is 1.0, lambda is 0.95
                        delta = R_[-j] + 1.0*v_n[-j] - v_[-j]
                        last_gae_lambda = delta + 1.0 * 0.95 * last_gae_lambda
                        adv_.append(last_gae_lambda)

                    adv_ = torch.stack(adv_).flip(0)
                    if len(adv_)>1: pass

                val_loss = torch.clamp(((v_-(R_+1.0*v_n))**2), -1e5, 1e5).mean()
                if args.R_as_adv:
                    adv_ = torch.clamp(R_, -1e5, 1e5)
                    val_loss*=0
                else:
                    adv_ = torch.clamp(adv_, -1e5, 1e5)

                rat_ = s_prob_g[:l_a-1]/(s_old_prob_g[:l_a-1])#+1e-5)
                pol_loss = torch.mean(
                            torch.clamp(torch.max(-adv_*rat_,
                                -adv_*torch.clamp(rat_, 1-0.2,1+0.2)), -1e5, 1e5)
                                )
                loss = loss + val_loss + pol_loss

            loss = loss / (i+1)
            if k < (args.ppo_k-1) and model.model_state.value != 'valid':
                if args.lr !=0:
                    self.opt.zero_grad()
                    torch.nn.utils.clip_grad_norm(self.parameters(), 5)
                    loss.backward()
                    self.opt.step()

            score_mean = score_mean / (i+1)
            score_std = score_std / (i+1)
            ll = len(score_comp)//2
            score_mean = (score_comp[:ll]-score_comp[ll:]).mean()
            score_std = (score_comp[:ll]-score_comp[ll:]).std()

        # K = 1
        if model.model_state.value != 'valid':
            score_all.append(score_mean.item())
            mean_kl_all.append(mean_kl)
            self.update_target_soft(tau=args.tau)
            np.save('training_score.npy', np.array(score_all)*args.ppo_k)
            np.save('training_mean_kl.npy', np.array(mean_kl_all))
            if self.counter % args.save_period==(1-args.save_period):
                torch.save(self.s_net.state_dict(), f'model/model_ethics_{self.counter}.bin')
            self.counter += 1
        if self.counter >= 3001:
            assert 1==2

            #self.counter += 1
            #if self.counter % 25 == 0:
            #    self.update_target()

        if args.lr == 0: loss*= 0
        return 0, loss, get_dict(loss, score_mean, score_std)



args = parse_args()
NUM_JOBS = 4
seed_everything(42)
score_all = []
mean_kl_all = []
train_df = pd.read_csv('data_nlg/train_simple.csv').head(100000)
valid_df = pd.read_csv('data_nlg/valid_simple.csv').head(10000)
#train_df = pd.read_csv('data/train_simple.csv').head(300000)
#valid_df = pd.read_csv('data/valid_simple.csv').sample(200)
train_df['text_id'] = np.arange(len(train_df))
valid_df['text_id'] = np.arange(len(valid_df))

tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join('tokenizer/emji_tokenizer/model.json'),
            bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
tok = tokenizer

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
    #model_name=args.model,
    num_train_steps=num_train_steps,
    learning_rate=max(args.lr, 1e-6),
    steps_per_epoch=len(train_dataset) / args.batch_size,
)

model = Tez(model)
es = EarlyStopping(
    monitor="train_ethics",
    model_path=os.path.join(f"model_test.bin"),
    patience=10000,
    mode="max",
    delta=0.0001,
    save_weights_only=True,
)
config = TezConfig(
    training_batch_size=args.batch_size,
    validation_batch_size=1,
    #validation_batch_size=args.valid_batch_size,
    gradient_accumulation_steps=args.accumulation_steps,
    epochs=args.epochs,
    #fp16=False,
    fp16=True,
    valid_shuffle=False,
    step_scheduler_after="batch",
    num_jobs=4,
    val_strategy="batch",
    val_steps=500000,
    clip_grad_norm=5,
    #val_steps=args.val_steps,
)


#def fit(self, train_dataset, valid_dataset=None, config: TezConfig = None, **kwargs):
model.fit(
    train_dataset, valid_dataset=valid_dataset,
    train_collate_fn=collate_fn, valid_collate_fn=collate_fn,
    config=config, callbacks=[es],
)
