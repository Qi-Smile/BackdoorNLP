import argparse
import os
import zipfile
from copy import copy
from typing import List, Tuple

import wget
import numpy as np
import stanza
import torch

from defense.onion import run_onion
from poison.models import load_model

from utils.data_utils import get_all_data, write_file
from utils.gpt2 import GPT2LM
   
parser = argparse.ArgumentParser()
parser.add_argument("--target-label", default=1, type=int)
parser.add_argument("--model-name", default="bert", help="albert, bert, roberta, lstm, distilbert")
parser.add_argument("--optimizer", default="adam", help="adam, sgd")
parser.add_argument("--lr", default=2e-5, help="1e-5, 1e-10, 2e-5", type=float)
parser.add_argument("--batch-size", default=2, type=int)
parser.add_argument("--dataset", default="ag", help="sst-2, ag, imbd")
parser.add_argument("--onion", default=False, help="defense")
parser.add_argument("--clean-data-path", default="/data/")
parser.add_argument("--output-path", default="/result_dataset/orderbkd_")
args = parser.parse_args()

DATASET_LINK = "https://nextcloud.ispras.ru/index.php/s/km9iNzswTC7gHS2/download/data.zip"
dataset_dir = "/data"
archive_name = "data.zip"
if not dataset_dir.exists():
    os.makedirs(exist_ok=True)
    wget.download(DATASET_LINK)
    with zipfile.ZipFile(archive_name) as zf:
        zf.extractall(dataset_dir)
    archive_name.unlink()

output_path = args.output_path + args.dataset + "/"
clean_data_path = args.clean_data_path + args.dataset + "/"
if not os.path.exists(output_path):
    os.makedirs(output_path + "poison_data/")
    os.makedirs(output_path + "clean_data/")
file = open(output_path + args.model_name + "_result.log", "w")