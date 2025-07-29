from typing import Union, List, Tuple, Sequence, Dict, Any, Optional
import logging
import numpy as np
import random
import torch

seed = 7

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

from torch import nn
from torch.utils.data import Dataset, DataLoader
import ankh
from transformers import Trainer, TrainingArguments, EvalPrediction
from datasets import load_dataset
import transformers.models.convbert as c_bert
from scipy import stats
from functools import partial
import pandas as pd
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class AnhkEncoder(object):
    def __init__(self,
        batch_size: int = 128,
        # progress_bar: bool = True
    ):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('Available device:', self.device)
        self.model, self.tokenizer = ankh.load_large_model()
        self.model.eval()
        self.model.to(device=self.device)
        # self.progress_bar = progress_bar
        self.batch_size = batch_size

    def preprocess_dataset(self, sequences, max_length=None):
        if max_length is None:
            max_length = len(max(sequences, key=lambda x: len(x)))
        splitted_sequences = [list(seq[:max_length]) for seq in sequences]
        return splitted_sequences

    def embed_dataset(self, model, sequences, shift_left=0, shift_right=-1):
        inputs_embedding = []
        with torch.no_grad():
            for sample in tqdm(sequences):
                ids = self.tokenizer.batch_encode_plus([sample], add_special_tokens=True,
                                                  padding=True, is_split_into_words=True,
                                                  return_tensors="pt")
                embedding = self.model(input_ids=ids['input_ids'].to(self.device))[0]
                embedding = embedding[0].detach().cpu().numpy()[shift_left:shift_right]
                inputs_embedding.append(embedding)
        return inputs_embedding

    def encode(self, sequences: [str]) -> np.ndarray:
        training_sequences = self.preprocess_dataset(sequences)
        encoding = self.embed_dataset(self.model, training_sequences)
        encoding = np.array(encoding)
        return encoding

# def encode(self, sequences: [str]) -> np.ndarray:

