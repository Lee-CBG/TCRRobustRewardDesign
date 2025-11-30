#!/usr/bin/env python

'''
python ensemble_bap.py --mode retraining --trainfile data/tcr_split/training.csv --bap cnn

CUDA_VISIBLE_DEVICES=1 python ensemble_bap.py --mode inference --testfile data/tcr_split/testing.csv --bap cnn
'''


import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from argparse import ArgumentParser

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

# Local imports
import utils
from nettcr_architectures import nettcr_one_chain, ergo_lstm

# Pandas display options (optional)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Argument parser
def parse_args():
    parser = ArgumentParser(description='TCR-Epitope Binding Predictor')
    parser.add_argument('--mode', choices=['retraining','finetuning','inference'], required=True,
                        help='Operation mode')
    parser.add_argument('--trainfile', help='Path to training CSV')
    parser.add_argument('--testfile', help='Path to testing CSV')
    parser.add_argument('--epochs', '-e', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--bap', choices=['cnn','lstm'], required=True,
                        help="Which BAP model: 'cnn' for netTCR CNN, 'lstm' for ERGO-LSTM")
    return parser.parse_args()

# Utility to get the correct column by possible names
def get_column(df, names):
    cols = [c for c in df.columns if c in names]
    if not cols:
        raise ValueError(f"Column not found. Options: {names}")
    if len(cols) > 1:
        raise ValueError(f"Multiple candidates: {cols}")
    return df[cols[0]].tolist()

# Main
if __name__ == '__main__':
    args = parse_args()

    # Common constants
    MAX_LEN = 22
    FEAT_DIM = 20
    peptide_cols = ['epi','Epitope','Epitopes','peptide']
    tcr_cols = ['tcr','TCR','TCRs']
    encoding = utils.blosum50_20aa

    # Load data as needed
    if args.mode in ('retraining','finetuning'):
        if not args.trainfile:
            sys.exit('Training file is required for training modes')
        train_df = pd.read_csv(args.trainfile)
        pep_train = utils.enc_list_bl_max_len(get_column(train_df, peptide_cols), encoding, MAX_LEN)
        tcr_train = utils.enc_list_bl_max_len(get_column(train_df, tcr_cols), encoding, MAX_LEN)
        y_train = train_df.binding.values
        train_inputs = [tcr_train, pep_train]

    if args.mode == 'inference':
        if not args.testfile:
            sys.exit('Test file is required for inference mode')
        test_df = pd.read_csv(args.testfile)
        pep_test = utils.enc_list_bl_max_len(get_column(test_df, peptide_cols), encoding, MAX_LEN)
        tcr_test = utils.enc_list_bl_max_len(get_column(test_df, tcr_cols), encoding, MAX_LEN)
        test_inputs = [tcr_test, pep_test]

    # Model selection
    if args.bap == 'cnn':
        model = nettcr_one_chain()
        ckpt_name = 'nettcr_scratch.hdf5'
    else:
        model = ergo_lstm()
        ckpt_name = 'ergo_lstm_scratch.hdf5'

    # Training
    if args.mode in ('retraining','finetuning'):
        ckpt_path = os.path.join('models', ckpt_name)
        callbacks = [
            ModelCheckpoint(ckpt_path, save_best_only=True, monitor='val_loss', mode='min'),
            EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1)
        ]
        model.compile(loss='binary_crossentropy', optimizer=Adam(1e-3))
        model.fit(
            x=train_inputs,
            y=y_train,
            validation_split=0.2,
            epochs=args.epochs,
            batch_size=128,
            callbacks=callbacks,
            verbose=1
        )

    # Inference
    elif args.mode == 'inference':
        ckpt_path = os.path.join('/mnt/disk11/user/pzhang84/recomb25/rewards/bap_ensemble/models', ckpt_name)
        model.load_weights(ckpt_path)
        preds = model.predict(test_inputs, verbose=0).ravel()
        
        # 转 logits
        eps = 1e-7
        p_clipped = np.clip(preds, eps, 1 - eps)      # 避免 log(0)
        logits = np.log(p_clipped) - np.log(1 - p_clipped)
        
        col = f'bap_{args.bap}'
        out_df = test_df.copy()
        out_df[col] = logits
        out_df.to_csv(args.testfile, index=False)

    else:
        sys.exit(f"Unknown mode: {args.mode}")
