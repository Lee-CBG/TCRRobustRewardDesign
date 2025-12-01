#!/usr/bin/env python

'''
python compute_embeds_tcrbert.py --input-file ./path/to/input.csv --output-file ./path/to/output.csv
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


def extract_embeds(tcr, tcr_mask, bert):
    with torch.inference_mode():
        tcr_outputs = bert(input_ids=tcr, attention_mask=tcr_mask, output_hidden_states=True)['hidden_states'][-1]
    # [B, L, C], [B, L] -> [B, L, C]
    embed_tcr = tcr_outputs.detach() * mask_cls_sep(tcr_mask).unsqueeze(-1)
    embed_tcr = embed_tcr.sum(dim=1) # [B, L, C] -> [B, C] 

    return embed_tcr



def tokenize_function(examples, tokenizer, max_length, tcr_key):
    tcr_with_spaces = [" ".join(list(seq_tcr)) for seq_tcr in examples[tcr_key]]
    tcr_outputs = tokenizer(tcr_with_spaces, truncation=True, max_length=max_length, padding="max_length", add_special_tokens=False)
    return {
        "input_ids_tcr": tcr_outputs["input_ids"],
        "attention_mask_tcr": tcr_outputs["attention_mask"],
    }


def trim_batch(input_ids, attention_mask):
    true_lengths = attention_mask.sum(dim=1)
    max_len = true_lengths.max()
    trimmed_input_ids = input_ids[:, :max_len]
    trimmed_attention_mask = attention_mask[:, :max_len]

    return trimmed_input_ids, trimmed_attention_mask

def get_key(df, names):
    cols = [c for c in df.columns if c in names]
    if not cols:
        raise ValueError(f"Column not found. Options: {names}")
    if len(cols) > 1:
        raise ValueError(f"Multiple candidates: {cols}")
    return cols[0]

def main():
    parser = argparse.ArgumentParser(description="Compute TCR-BERT TCR embeds")

    parser.add_argument('--input-path', type=str, required=True,
                        help='The path to the input file or directory.')
    parser.add_argument('--output-path', type=str, required=True,
                        help='The path for the output file.')

    # Parse the arguments
    args = parser.parse_args()
    
    df = pd.read_csv(args.input_path)
    
    eval_bs = 128

    tcr_cols = ['tcr','TCR','TCRs']
    tcr_key = get_key(df, tcr_cols)
    
    max_length = int(df[tcr_key].str.len().max())
    print(max_length)
    mask = df[tcr_key].apply(filter_tcr)
    df_filtered = df[mask]
    ds = datasets.Dataset.from_pandas(df_filtered)
    
    model, tokenizer = load_BERT_model_and_tokenizer("wukevin/tcr-bert", "wukevin/tcr-bert", device)
    tokenized_dataset = ds.map(partial(tokenize_function, tokenizer=tokenizer, max_length=max_length, tcr_key=tcr_key), batched=True, remove_columns=[tcr_key], num_proc=4)
    data_collator = DefaultDataCollator(return_tensors='pt')
    
    eval_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=eval_bs,
        collate_fn=data_collator,
        num_workers=3,
        drop_last=False,
    )
    
    model.to(device)
    model.eval()
    
    all_embeds = []
    
    for i, batch in enumerate(tqdm(eval_dataloader)):
        with torch.inference_mode():
            batch = {k: v.to(device) for k, v in batch.items()}
            batch['input_ids_tcr'], batch['attention_mask_tcr'] = trim_batch(batch['input_ids_tcr'], batch['attention_mask_tcr'])
    
            embeds = extract_embeds(batch['input_ids_tcr'], batch['attention_mask_tcr'], model)
            all_embeds.extend(embeds.cpu().tolist())
            
    df.loc[mask, 'embeds'] = pd.Series(all_embeds, index=df.index[mask])
    df.to_csv(args.output_path, index=False)




if __name__ == '__main__':
    main()
