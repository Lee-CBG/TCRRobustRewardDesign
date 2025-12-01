#!/usr/bin/env python

'''
python catelmo_bap_infer.py --input-path ./path/to/inputs.csv --output-path ./path/to/outputs.csv --model-path ./path/to/model.pth
'''

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer, DefaultDataCollator
from torch.utils.data import DataLoader
import datasets
import sys
import argparse
from functools import partial
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
def filter_tcr(tcr):
    if tcr == 'WRONGFORMAT':
        return False
    if not isinstance(tcr, str):
        return False
    if not tcr.isalpha():
        return False
    return True
    

class ELMo(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, rnn_cls=nn.LSTM):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # TODO check if can use bidirectional=True
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.fw_rnn = nn.ModuleList([
            rnn_cls(hidden_size, hidden_size, batch_first=True) for _ in range(num_layers)
        ])
        self.bw_rnn = nn.ModuleList([
            rnn_cls(hidden_size, hidden_size, batch_first=True) for _ in range(num_layers)
        ])

    def forward_features(self, x):
        # Not optimal for inference/downstream throughput but does it matter if we keep bs high?
        # [B, L] token ids
        x = self.embedding(x)  # [B, L, C]

        fw_outputs = []
        bw_outputs = []
        x_fw = x
        x_bw = x.flip(1)
        state_fw = None
        state_bw = None
        for fw_rnn, bw_rnn in zip(self.fw_rnn, self.bw_rnn):
            x_fw_new, _ = fw_rnn(x_fw)
            x_fw = x_fw_new + x_fw
            # FIXME properly mask off suffix of sequence for bw direction by resetting the state when attn_mask is 0 for a given position
            x_bw_new, _ = bw_rnn(x_bw)
            x_bw = x_bw_new + x_bw
            fw_outputs.append(x_fw)
            bw_outputs.append(x_bw.flip(1))
        logits_fw = x_fw @ self.embedding.weight.T  # [B, L, C] @ [C, V] -> [B, L, V]
        logits_bw = x_bw.flip(1) @ self.embedding.weight.T  # [B, L, C] @ [C, V] -> [B, L, V]
        return x, logits_fw, logits_bw, fw_outputs, bw_outputs

    def forward_loss(self, x, loss_mask = None):
        _, logits_fw, logits_bw, _, _ = self.forward_features(x)

        preds_fw = logits_fw[:, :-1, :].reshape(-1, self.vocab_size)
        targets_fw = x[:, 1:].reshape(-1)
        loss_fw = F.cross_entropy(preds_fw, targets_fw, reduction="none")

        preds_bw = logits_bw[:, 1:, :].reshape(-1, self.vocab_size)
        targets_bw = x[:, :-1].reshape(-1)
        loss_bw = F.cross_entropy(preds_bw, targets_bw, reduction="none")


        if loss_mask is not None:
            mask_fwd = loss_mask[:, 1:].reshape(-1)
            mask_bwd = loss_mask[:, :-1].reshape(-1)
            loss = (loss_fw*mask_fwd + loss_bw*mask_bwd)
            valid = mask_fwd.sum() + mask_bwd.sum()
            loss = loss.sum() / valid
        else:
            loss = loss_fw + loss_bw
            loss = loss.mean()

        return loss

class MlpHead(nn.Module):
    def __init__(
            self,
            in_features,
            other_features,
            hidden_features,
            out_features,
            act_layer=nn.GELU,
            bias=True,
            drop=0.1,
    ):
        super().__init__()

        self.fc1 = nn.Linear(in_features+other_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
    def forward(self, x, other):
        x = torch.cat((x, other), dim=-1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return x

def mask_cls_sep(attention_mask):
    # clone mask
    modified_mask = attention_mask.clone()

    # mask [CLS]
    modified_mask[:, 0] = 0

    # find location of [SEP] for each item in batch
    # sep_idx = seq_len - 1
    sequence_lengths = torch.sum(attention_mask, dim=1)

    for i in range(attention_mask.shape[0]):
        sep_token_index = sequence_lengths[i] - 1
        # Ensure the index is valid before modifying
        if sep_token_index >= 0:
            modified_mask[i, sep_token_index] = 0

    return modified_mask

class BAPHead(nn.Module):
    def __init__(self, n_layers, dim, hidden_dim=None):
        super().__init__()
        self.weights_epi = nn.Parameter(torch.zeros(2*n_layers + 1))
        self.weights_tcr = nn.Parameter(torch.zeros(2*n_layers + 1))
        self.head = MlpHead(
            dim,
            dim,
            hidden_dim or dim,
            1,
        )
    def forward(self, epi, epi_mask, tcr, tcr_mask, elmo):
        noise_aug_str = 0.3
        # [B, L, C], N*[B, L, C], N*[B, L, C]
        with torch.inference_mode():
            embed, _, _, fw_outputs, bw_outputs = elmo.forward_features(epi)
            
        with torch.no_grad():
            epi_outputs = torch.stack([embed, *fw_outputs, *bw_outputs], dim=0)
            if self.training:
                epi_noise = torch.randn_like(epi_outputs) * noise_aug_str * epi_outputs.std(dim=(1,2), keepdim=True)
                epi_outputs = epi_outputs + epi_noise
                
        embed_epi = epi_outputs.detach() * (epi_mask.unsqueeze(0).unsqueeze(-1) * self.weights_epi.softmax(dim=0).reshape(-1, 1, 1, 1))
        embed_epi = embed_epi.sum(dim=(0,2))
        
        with torch.inference_mode():
            embed, _, _, fw_outputs, bw_outputs = elmo.forward_features(tcr)
            
        with torch.no_grad():
            tcr_outputs = torch.stack([embed, *fw_outputs, *bw_outputs], dim=0)
            if self.training:
                tcr_noise = torch.randn_like(tcr_outputs) * noise_aug_str * tcr_outputs.std(dim=(1,2), keepdim=True)
                tcr_outputs = tcr_outputs + tcr_noise
        
        embed_tcr = tcr_outputs.detach() * (tcr_mask.unsqueeze(0).unsqueeze(-1) * self.weights_tcr.softmax(dim=0).reshape(-1, 1, 1, 1))
        embed_tcr = embed_tcr.sum(dim=(0,2))

        logits = self.head(embed_epi, embed_tcr)
        return logits



def tokenize_function_bap(examples, tokenizer, max_length):
    epi_with_spaces = [" ".join(list(seq_tcr)) for seq_tcr in examples['Epitopes']]
    epi_outputs = tokenizer(epi_with_spaces, truncation=True, max_length=max_length, padding="max_length", add_special_tokens=False)
    tcr_with_spaces = [" ".join(list(seq_tcr)) for seq_tcr in examples['TCRs']]
    tcr_outputs = tokenizer(tcr_with_spaces, truncation=True, max_length=max_length, padding="max_length", add_special_tokens=False)
    return {
        "input_ids_epi": epi_outputs["input_ids"],
        "attention_mask_epi": epi_outputs["attention_mask"],
        "input_ids_tcr": tcr_outputs["input_ids"],
        "attention_mask_tcr": tcr_outputs["attention_mask"],
    }


def trim_batch(input_ids, attention_mask):
    true_lengths = attention_mask.sum(dim=1)
    max_len = true_lengths.max()
    trimmed_input_ids = input_ids[:, :max_len]
    trimmed_attention_mask = attention_mask[:, :max_len]

    return trimmed_input_ids, trimmed_attention_mask

def main():
    parser = argparse.ArgumentParser(description="Run inference on TCR-BERT BAP heads")

    parser.add_argument('--input-path', type=str, required=True,
                        help='The path to the input file or directory.')
    parser.add_argument('--output-path', type=str, required=True,
                        help='The path for the output file.')
    parser.add_argument('--model-path', type=str, required=True,
                        help='The path to the model file.')

    # Parse the arguments
    args = parser.parse_args()
    
    df = pd.read_csv(args.input_path)
    
    eval_bs = 128
    
    max_length = int(max(df['TCRs'].str.len().max(), df['Epitopes'].str.len().max()))
    print(max_length)
    mask = df['TCRs'].apply(filter_tcr)
    df_filtered = df[mask]
    ds = datasets.Dataset.from_pandas(df_filtered)
    
    tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ELMo(tokenizer.vocab_size, 1024, num_layers=12, rnn_cls = nn.LSTM)

    model.load_state_dict(torch.load("./rewards/catelmo_reimpl_20M.pth", map_location='cpu'))

    tokenized_dataset = ds.map(partial(tokenize_function_bap, tokenizer=tokenizer, max_length=max_length), batched=True, remove_columns=['Epitopes', 'TCRs'], num_proc=4)
    data_collator = DefaultDataCollator(return_tensors='pt')
    
    eval_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=eval_bs,
        collate_fn=data_collator,
        num_workers=3,
        drop_last=False,
    )
    
    classifier = BAPHead(12, 1024, hidden_dim=2048)
    classifier.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.to(device)
    model.eval()
    classifier = classifier.to(device)
    classifier.eval()
    
    all_logits = []
    
    for i, batch in enumerate(tqdm(eval_dataloader)):
        with torch.inference_mode():
            batch = {k: v.to(device) for k, v in batch.items()}
            batch['input_ids_epi'], batch['attention_mask_epi'] = trim_batch(batch['input_ids_epi'], batch['attention_mask_epi'])
            batch['input_ids_tcr'], batch['attention_mask_tcr'] = trim_batch(batch['input_ids_tcr'], batch['attention_mask_tcr'])
    
            logits = classifier(batch['input_ids_epi'], batch['attention_mask_epi'], batch['input_ids_tcr'], batch['attention_mask_tcr'], model)
            all_logits.extend(logits[:,0].cpu().tolist())
            
    df['logits'] = float(-10) # default value for invalid TCRs
    df.loc[mask, 'logits'] = all_logits
    df.to_csv(args.output_path, index=False)




if __name__ == '__main__':
    main()