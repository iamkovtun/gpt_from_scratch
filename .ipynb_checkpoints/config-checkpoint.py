import torch
from pathlib import Path
import os
# +

def get_config():
     return {"num_of_epoches": 20,
              "lr": 10**-4,
              "d_model": 512,
              "model_folder": "weights",
              "model_basename": "tmodel_",
              "preload": "17",
              "tokenizer_file": "tokenizer_{0}.json",
              "dataset_name": "opus_books",
              "lang_src": "en",
              "lang_tgt": "it",
              "seq_len": 100,
              "auto_drop": True,
              "batch_size": 64,
              "experiment_name": "tmodel",
              "head_numbers": 8,
              "dropout": 0.1,
              "dff": 2048,
              "N": 6,
              }


# -

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['dataset_name']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)


def get_latest_weight(config):
    model_folder = f"{config['dataset_name']}_{config['model_folder']}"
    # Make a list with all files' names in model_folder
    files = sorted(os.listdir(str(Path('.') / model_folder)))
    # Get the last file name (latest file based on naming convention)
    if files == None:
        return None
    model_filename = files[-1]
    # Return the full path of the file
    return str(Path('.') / model_folder / model_filename)






