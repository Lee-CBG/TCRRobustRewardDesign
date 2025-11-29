'''
python nettcr.py --mode retraining --trainfile data/tcr_split/training.csv --testfile data/tcr_split/testing.csv

python nettcr.py --mode finetuning --trainfile ../attack_netTCR/manual_attack_kl_co_0.8/iter_2/data/trainData.csv --testfile ../attack_netTCR/manual_attack_kl_co_0.8/iter_2/data/testData.csv

python nettcr.py --mode inference --model_name nettcr_finetuned_iter_2.hdf5 --trainfile ../attack_netTCR/manual_attack/iter_1/data/trainData.csv --testfile ../attack_netTCR/manual_attack/iter_2/data/testData.csv

python nettcr.py --mode inference --model_name nettcr_finetuned_iter_2.hdf5 --trainfile ../attack_netTCR/manual_attack/iter_1/data/trainData.csv --testfile ../NetTCR/data/tcr_split/testing.csv
'''


import os, sys
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
import keras
import time
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D, concatenate
from tensorflow.keras.optimizers import Adam
from keras.initializers import glorot_normal
from keras.activations import sigmoid
from sklearn.metrics import roc_auc_score
import utils
import keras.backend as K
from keras.callbacks import EarlyStopping

from nettcr_architectures import ergo_lstm, nettcr_one_chain 

#Options for Pandas DataFrame printing
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', -1)

from argparse import ArgumentParser

#Args parser 
parser = ArgumentParser(description="Specifying Input Parameters")
parser.add_argument("-tr", "--trainfile", help="Specify the full path of the training file with TCR sequences")
parser.add_argument("-te", "--testfile", help="Specify the full path of the file with TCR sequences")
parser.add_argument("-mode", "--mode", default="retraining", help="retraining or finetuning")
parser.add_argument("-model_name", "--model_name", help="nettcr_scratch.hdf5")
parser.add_argument("-c", "--chain", default="b", help="Specify the chain(s) to use (a, b, ab). Default: ab")
parser.add_argument("-o", "--outfile", default=sys.stdout, help="Specify output file")
parser.add_argument("-e", "--epochs", default=200, type=int, help="Specify the number of epochs")

parser.add_argument(
    "--bap", choices=["cnn","lstm"], required=True,
    help="Which BAP model to use: 'cnn' for nettcr_one_chain or 'lstm' for ERGO-LSTM"
)

args = parser.parse_args()

EPOCHS = int(args.epochs)
chain = args.chain


train_data = pd.read_csv(args.trainfile)
test_data = pd.read_csv(args.testfile)


# Encode data
encoding = utils.blosum50_20aa


def get_column(df, names):
    """Retrieves a column, trying multiple names. Raises ValueError if not found or multiple found."""
    found = [col for col in df.columns if col in names]
    if not found:
        raise ValueError(f"Column not found. Please provide one of: {names}")
    if len(found) > 1:
        raise ValueError(f"Multiple matching columns found: {found}. Use only one of: {names}")
    return df[found[0]].tolist()  # Return as a list directly

# Call and compile the model

peptide_names = ["epi", "Epitope", "Epitopes", "peptide"]
tcr_names = ["tcr", "TCR", "TCRs"]

try: # wrap in try except block
    pep_train = utils.enc_list_bl_max_len(get_column(train_data, peptide_names), encoding, 22)
    tcrb_train = utils.enc_list_bl_max_len(get_column(train_data, tcr_names), encoding, 22)
    pep_test = utils.enc_list_bl_max_len(get_column(test_data, peptide_names), encoding, 22)
    tcrb_test = utils.enc_list_bl_max_len(get_column(test_data, tcr_names), encoding, 22)
except ValueError as e: # catch the error and print the message
    print(f"Error processing data: {e}")
    # handle error, e.g. exit the function


y_train = np.array(train_data.binding) # This line is unchanged

train_inputs = [tcrb_train, pep_train]
test_inputs = [tcrb_test, pep_test]

if args.bap == 'cnn':
    mdl = nettcr_one_chain()
elif args.bap == 'lstm':
    mdl = ergo_lstm()
    
    
    
if args.mode == 'retraining':
    checkpoint_filepath = f'models/ergo_lstm_scratch.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )
    es = EarlyStopping(monitor='val_loss', mode='min', patience=30, verbose=1)


    mdl.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.001))
    history = mdl.fit(train_inputs, y_train, validation_split=0.2,
                      epochs=EPOCHS, batch_size=128, verbose=1, callbacks=[es, model_checkpoint_callback])

    
    

if args.mode == 'inference':
    
    if args.bap == 'cnn':
        model_name = 'nettcr_scratch.hdf5'
        col = 'bap_cnn'
    elif args.bap == 'lstm':
        model_name = 'ergo_lstm_scratch.hdf5'
        col = 'bap_lstm'
        
    # Load the pre-trained model weights
    checkpoint_filepath = f'/mnt/disk11/user/pzhang84/recomb25/rewards/bap_ensemble/models/{model_name}'
    mdl.load_weights(checkpoint_filepath)
    
    preds = mdl.predict(test_inputs, verbose=0)
    pred_df = pd.concat([test_data,
                         pd.Series(np.ravel(preds), name=col)],
                        axis=1)
    pred_df.to_csv(args.testfile, index=False)
