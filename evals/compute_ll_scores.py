#!/usr/bin/env python

'''
python compute_ll_scores.py --input-path /path/to/input/dir --output-path /path/to/output/dir
'''

import torch
import torch.nn.functional as F
import pandas as pd
import json
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
from copy import deepcopy
import os
import argparse
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

def load_BERT_model_and_tokenizer(model_dir, tokenizer_dir, device):
    model = AutoModelForMaskedLM.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    model.to(device)
    return model, tokenizer

def load_GPT_model_and_tokenizer(model_dir, tokenizer_dir, device):
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    model.to(device)
    return model, tokenizer

# epi-tcr pairs
def tokenize_function(examples, tokenizer):
    # The tokenizer gives [CLS] <input> [SEP]
    # We add a space between each amino acid character to treat them as separate "words".
    sequences_with_spaces = [" ".join(list(seq_epi) + [tokenizer.sep_token] + list(seq_tcr)) for seq_epi, seq_tcr in zip(examples['epi'], examples['tcr'])]
    tokenizer_outputs = tokenizer(sequences_with_spaces, return_tensors='pt')
    return tokenizer_outputs

def tokenize_masking(epi, tcr, tokenizer):
    sequence = list(epi) + [tokenizer.sep_token] + list(tcr)
    masked_sequences = []
    for i in range(len(tcr)):
        masked_sequences.append(deepcopy(sequence))
        masked_sequences[i][len(epi) + 1 + i] = tokenizer.mask_token
        masked_sequences[i] = " ".join(masked_sequences[i])
    return tokenizer(masked_sequences, return_tensors='pt')


def tokenize_function_tcr_only(examples, tokenizer):
    # The tokenizer gives [CLS] <input> [SEP]
    # We add a space between each amino acid character to treat them as separate "words".
    sequences_with_spaces = [" ".join(list(seq_tcr)) for seq_tcr in examples['tcr']]
    tokenizer_outputs = tokenizer(sequences_with_spaces, return_tensors='pt')
    return tokenizer_outputs

def tokenize_function_gpt(examples, tokenizer):
    sequences_with_spaces = ["".join(['<tcr>'] + list(seq_tcr)) for seq_tcr in examples['tcr']]
    tokenizer_outputs = tokenizer(sequences_with_spaces, return_tensors='pt')
    return tokenizer_outputs

def tokenize_masking_tcr_only(tcr, tokenizer):
    sequence = list(tcr)
    masked_sequences = []
    for i in range(len(tcr)):
        masked_sequences.append(deepcopy(sequence))
        masked_sequences[i][i] = tokenizer.mask_token
        masked_sequences[i] = " ".join(masked_sequences[i])
    return tokenizer(masked_sequences, return_tensors='pt')

def compute_log_likelihood_rewrite(epi, generated_tcr, model, tokenizer, device):
    if not filter_tcr(generated_tcr):
        print(f"got bad tcr: {generated_tcr}")
        return 0.0
    encoded_input = tokenize_function({'epi': [epi], 'tcr': [generated_tcr]}, tokenizer)
    encoded_input = {key: value.to(device) for key, value in encoded_input.items()}
    with torch.no_grad():
        outputs = model(**encoded_input)
    logits = F.log_softmax(outputs.logits, dim=-1)
    total_log_likelihood = 0.0
    sequence_length = len(generated_tcr)
    for i, amino_acid in enumerate(generated_tcr):
        token_id = tokenizer.convert_tokens_to_ids(amino_acid)
        if token_id == tokenizer.unk_token_id:
            continue
        log_likelihood = logits[0, i + len(epi) + 2, token_id]
        total_log_likelihood += log_likelihood.item()
    average_log_likelihood = total_log_likelihood / sequence_length
    return average_log_likelihood

def compute_log_likelihood_masking(epi, generated_tcr, model, tokenizer, device):
    if not filter_tcr(generated_tcr):
        print(f"got bad tcr: {generated_tcr}")
        return 0.0
    encoded_input = tokenize_masking(epi, generated_tcr, tokenizer)
    encoded_input = {key: value.to(device) for key, value in encoded_input.items()}
    with torch.no_grad():
        outputs = model(**encoded_input)
    logits = F.log_softmax(outputs.logits, dim=-1)
    total_log_likelihood = 0.0
    sequence_length = len(generated_tcr)
    for i, amino_acid in enumerate(generated_tcr):
        token_id = tokenizer.convert_tokens_to_ids(amino_acid)
        if token_id == tokenizer.unk_token_id:
            continue
        log_likelihood = logits[i, i + len(epi) + 2, token_id]
        total_log_likelihood += log_likelihood.item()
    average_log_likelihood = total_log_likelihood / sequence_length
    return average_log_likelihood

def compute_log_likelihood_rewrite_tcr_only(generated_tcr, model, tokenizer, device):
    if not filter_tcr(generated_tcr):
        print(f"got bad tcr: {generated_tcr}")
        return 0.0
    encoded_input = tokenize_function_tcr_only({'tcr': [generated_tcr]}, tokenizer)
    encoded_input = {key: value.to(device) for key, value in encoded_input.items()}
    with torch.no_grad():
        outputs = model(**encoded_input)
    logits = F.log_softmax(outputs.logits, dim=-1)
    total_log_likelihood = 0.0
    sequence_length = len(generated_tcr)
    for i, amino_acid in enumerate(generated_tcr):
        token_id = tokenizer.convert_tokens_to_ids(amino_acid)
        if token_id == tokenizer.unk_token_id:
            continue
        log_likelihood = logits[0, i + 1, token_id]
        total_log_likelihood += log_likelihood.item()
    average_log_likelihood = total_log_likelihood / sequence_length
    return average_log_likelihood

def compute_log_likelihood_masking_tcr_only(generated_tcr, model, tokenizer, device):
    if not filter_tcr(generated_tcr):
        print(f"got bad tcr: {generated_tcr}")
        return 0.0
    encoded_input = tokenize_masking_tcr_only(generated_tcr, tokenizer)
    encoded_input = {key: value.to(device) for key, value in encoded_input.items()}
    with torch.no_grad():
        outputs = model(**encoded_input)
    logits = F.log_softmax(outputs.logits, dim=-1)
    total_log_likelihood = 0.0
    sequence_length = len(generated_tcr)
    for i, amino_acid in enumerate(generated_tcr):
        token_id = tokenizer.convert_tokens_to_ids(amino_acid)
        if token_id == tokenizer.unk_token_id:
            continue
        log_likelihood = logits[i, i + 1, token_id]
        total_log_likelihood += log_likelihood.item()
    average_log_likelihood = total_log_likelihood / sequence_length
    return average_log_likelihood

def compute_log_likelihood_gpt(generated_tcr, model, tokenizer, device):
    if not filter_tcr(generated_tcr):
        print(f"got bad tcr: {generated_tcr}")
        return 0.0
    encoded_input = tokenize_function_gpt({'tcr': [generated_tcr]}, tokenizer)
    encoded_input = {key: value.to(device) for key, value in encoded_input.items()}
    with torch.no_grad():
        outputs = model(**encoded_input)
    logits = F.log_softmax(outputs.logits, dim=-1)
    total_log_likelihood = 0.0
    targets = tokenizer.batch_decode(encoded_input['input_ids'])[0].split(" ")
    sequence_length = len(targets) - 1
    if sequence_length == 0:
        return 0.0
    # match prediction @ i with actual @ i+1
    for i, amino_acid in enumerate(targets[1:]):
        token_id = tokenizer.convert_tokens_to_ids(amino_acid)
        if token_id == tokenizer.unk_token_id:
            continue
        log_likelihood = logits[0, i, token_id]
        total_log_likelihood += log_likelihood.item()
    average_log_likelihood = total_log_likelihood / sequence_length
    return average_log_likelihood

def compute_tokenwise_log_likelihoods_rewrite(epi, generated_tcr, model, tokenizer, device):
    encoded_input = tokenize_function({'epi': [epi], 'tcr': [generated_tcr]}, tokenizer)
    encoded_input = {key: value.to(device) for key, value in encoded_input.items()}
    with torch.no_grad():
        outputs = model(**encoded_input)
    logits = outputs.logits[:, len(epi)+2:-1, :]  # [1, seq_len, vocab_size]
    token_ids = encoded_input['input_ids'][0][len(epi)+2:-1]  # [seq_len]
    log_probs = torch.log_softmax(logits, dim=-1)  # [1, seq_len, vocab_size]
    token_log_likelihoods = log_probs[0, torch.arange(len(token_ids)), token_ids]  # [seq_len]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    return tokens, token_log_likelihoods.cpu().tolist()

def compute_tokenwise_log_likelihoods_masking(epi, generated_tcr, model, tokenizer, device):
    encoded_input = tokenize_masking(epi, generated_tcr, tokenizer)
    encoded_input = {key: value.to(device) for key, value in encoded_input.items()}
    token_ids = tokenizer(" ".join(list(generated_tcr)), return_tensors='pt')['input_ids'][0][1:-1].to(device)
    with torch.no_grad():
        outputs = model(**encoded_input)
    logits = torch.diagonal(outputs.logits, len(epi)+2).transpose(0, 1).unsqueeze(0)
    #token_ids = torch.diagonal(encoded_input['input_ids'], len(epi)+2)
    log_probs = torch.log_softmax(logits, dim=-1)  # [1, seq_len, vocab_size]
    token_log_likelihoods = log_probs[0, torch.arange(len(token_ids)), token_ids]  # [seq_len]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    return tokens, token_log_likelihoods.cpu().tolist()

def compute_log_likelihood_batch(tcr_list, model, tokenizer, device):
    scores = []
    for tcr in tcr_list:
        if isinstance(tcr, str) and tcr != 'WRONGFORMAT':
            score = compute_log_likelihood(tcr, model, tokenizer, device)
        else:
            score = -10.0
        scores.append(score)
    return scores

def get_key(df, names):
    cols = [c for c in df.columns if c in names]
    if not cols:
        raise ValueError(f"Column not found. Options: {names}")
    if len(cols) > 1:
        raise ValueError(f"Multiple candidates: {cols}")
    return cols[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute log-likelihood scores")
    parser.add_argument('--input-path', type=str, required=True,
                        help='The path to the input directory.')
    parser.add_argument('--output-path', type=str, required=True,
                        help='The path for the output directory.')
    args = parser.parse_args()

    entries = os.listdir(args.input_path)
    files = [f for f in entries if os.path.isfile(os.path.join(args.input_path, f))]
    
    print("Got the following files:")
    print(files)

    #model, tokenizer = load_BERT_model_and_tokenizer("/content/drive/MyDrive/LEE-CBG/TCR_MLM_Analyze/model", "Rostlab/prot_bert_bfd", device)
    #model_tcr_only, tokenizer_tcr_only = load_BERT_model_and_tokenizer("Rostlab/prot_bert_bfd", "Rostlab/prot_bert_bfd", device)
    model_tcrbert, tokenizer_tcrbert = load_BERT_model_and_tokenizer("wukevin/tcr-bert", "wukevin/tcr-bert", device)
    model_tcrbert_mlm, tokenizer_tcrbert_mlm = load_BERT_model_and_tokenizer("wukevin/tcr-bert-mlm-only", "wukevin/tcr-bert-mlm-only", device)
    model_gpt, tokenizer_gpt = load_GPT_model_and_tokenizer("./rita_m_sft_checkpoint-6400", "lightonai/RITA_m", device)
    special_tokens_dict = {'eos_token': '<EOS>', 'pad_token': '<PAD>', 'additional_special_tokens': ['$','<tcr>']}
    num_added_toks = tokenizer_gpt.add_special_tokens(special_tokens_dict)
    
    for file_path in files:
        if os.path.isdir(args.input_path):
            df = pd.read_csv(os.path.join(args.input_path, file_path))
        else:
            df = pd.read_csv(args.input_path)
        bert_ll_all = []
        bert_ll_masking = []
        tcrbert_ll_all = []
        tcrbert_ll_masking = []
        tcrbert_mlm_ll_all = []
        tcrbert_mlm_ll_masking = []
        sft_ll_all = []
        sft_ll_masking = []
        gpt_ll = []
        peptide_cols = ['epi','Epitope','Epitopes','peptide']
        tcr_cols = ['tcr','TCR','TCRs']
        tcr_key = get_key(df, tcr_cols)
        #epi_key = get_key(df, epi_cols)
        for index, row in tqdm(df.iterrows()):
            #bert_ll_all.append(compute_log_likelihood_rewrite_tcr_only(row[tcr_key], model_tcr_only, tokenizer_tcr_only, device))
            #bert_ll_masking.append(compute_log_likelihood_masking_tcr_only(row[tcr_key], model_tcr_only, tokenizer_tcr_only, device))
            tcrbert_ll_all.append(compute_log_likelihood_rewrite_tcr_only(row[tcr_key], model_tcrbert, tokenizer_tcrbert, device))
            tcrbert_ll_masking.append(compute_log_likelihood_masking_tcr_only(row[tcr_key], model_tcrbert, tokenizer_tcrbert, device))
            tcrbert_mlm_ll_all.append(compute_log_likelihood_rewrite_tcr_only(row[tcr_key], model_tcrbert_mlm, tokenizer_tcrbert_mlm, device))
            tcrbert_mlm_ll_masking.append(compute_log_likelihood_masking_tcr_only(row[tcr_key], model_tcrbert_mlm, tokenizer_tcrbert_mlm, device))
            #sft_ll_all.append(compute_log_likelihood_rewrite(row[epi_key], row[tcr_key'], model, tokenizer, device))
            #sft_ll_masking.append(compute_log_likelihood_masking(row[epi_key], row[tcr_key], model, tokenizer, device))
            gpt_ll.append(compute_log_likelihood_gpt(row[tcr_key], model_gpt, tokenizer_gpt, device))
        #df['bert_ll_all'] = bert_ll_all
        #df['bert_ll_masking'] = bert_ll_masking
        df['tcrbert_ll_all'] = tcrbert_ll_all
        df['tcrbert_ll_masking'] = tcrbert_ll_masking
        df['tcrbert_mlm_ll_all'] = tcrbert_mlm_ll_all
        df['tcrbert_mlm_ll_masking'] = tcrbert_mlm_ll_masking
        #df['sft_ll_all'] = sft_ll_all
        #df['sft_ll_masking'] = sft_ll_masking
        df['gpt_ll'] = gpt_ll

        os.makedirs(args.output_path, exist_ok=True)
        df.to_csv(os.path.join(args.output_path, file_path), index=False)




