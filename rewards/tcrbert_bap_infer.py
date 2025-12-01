#!/usr/bin/env python

'''
python tcrbert_bap_infer.py --input-path ./path/to/inputs.csv --output-path ./path/to/outputs.csv --model-path ./path/to/model.pth
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

def load_BERT_model_and_tokenizer(model_dir, tokenizer_dir, device):
    model = AutoModelForMaskedLM.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    model.to(device)
    return model, tokenizer
    
def filter_tcr(tcr):
    if tcr == 'WRONGFORMAT':
        return False
    if not isinstance(tcr, str):
        return False
    if not tcr.isalpha():
        return False
    return True

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
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        head_cls = MlpHead
        self.head = head_cls(
            dim,
            dim,
            hidden_dim or dim,
            1,
        )
    def forward(self, epi, epi_mask, tcr, tcr_mask, bert):
        # [B, L, C], N*[B, L, C], N*[B, L, C]
        with torch.inference_mode():
            epi_outputs = bert(input_ids=epi, attention_mask=epi_mask, output_hidden_states=True)['hidden_states']
        with torch.no_grad():
            epi_outputs = torch.stack(epi_outputs, dim=0)
            
        embed_epi = epi_outputs.detach() * mask_cls_sep(epi_mask).unsqueeze(0).unsqueeze(-1)
        embed_epi = embed_epi.sum(dim=(0,2))

        with torch.inference_mode():
            tcr_outputs = bert(input_ids=tcr, attention_mask=tcr_mask, output_hidden_states=True)['hidden_states']
        with torch.no_grad():
            tcr_outputs = torch.stack(tcr_outputs, dim=0)

        embed_tcr = tcr_outputs.detach() * mask_cls_sep(tcr_mask).unsqueeze(0).unsqueeze(-1)
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

    parser.add_argument('--input-path', type=str, default='./tmp_epis_tcrs.csv',
                        help='The path to the input file or directory.')
    parser.add_argument('--output-path', type=str, default='./tmp_epis_tcrs_1.csv',
                        help='The path for the output file.')
    parser.add_argument('--model-path', type=str, default='./tcrbert_bap_head_robust.pth',
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
    
    model, tokenizer = load_BERT_model_and_tokenizer("wukevin/tcr-bert", "wukevin/tcr-bert", device)
    tokenized_dataset = ds.map(partial(tokenize_function_bap, tokenizer=tokenizer, max_length=max_length), batched=True, remove_columns=['Epitopes', 'TCRs'], num_proc=4)
    data_collator = DefaultDataCollator(return_tensors='pt')
    
    eval_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=eval_bs,
        collate_fn=data_collator,
        num_workers=3,
        drop_last=False,
    )
    
    classifier = BAPHead(768, hidden_dim=2048)
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