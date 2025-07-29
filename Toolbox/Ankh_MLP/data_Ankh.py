import warnings
import numpy as np
import pandas as pd
from Bio import SeqIO
import torch.utils.data
from sklearn.model_selection import KFold, ShuffleSplit

from global_feature import AnhkEncoder
import torch

class SequenceData(torch.utils.data.Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sequences[index], self.labels[index]


class MetagenesisData(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    

class Dataset(object):
    def __init__(self,
            train_tsv=None, test_tsv=None,
            fasta=None):
        """
        split_ratio: [train, valid] or [train, valid, test]
        """

        self.train_tsv = train_tsv
        self.test_tsv = test_tsv
        self.fasta = fasta
    
        self.native_sequence = self._read_native_sequence()
        if train_tsv is not None:
            self.full_df = self._read_mutation_df(train_tsv)
        else:
            self.full_df = None

        if test_tsv is not None:
            self.test_df = self._read_mutation_df(test_tsv)
            
        self.anhk_encoder = AnhkEncoder()
            
    def _read_native_sequence(self):
        fasta = SeqIO.read(self.fasta, 'fasta')
        native_sequence = str(fasta.seq)
        return native_sequence

    def _mutation_to_sequence(self, mutation):
        '''
        Parameters
        ----------
        mutation: ';'.join(WiM) (wide-type W at position i mutated to M)
        '''
        sequence = self.native_sequence
        for mut in mutation.split(';'):
            wt_aa = mut[0]
            mt_aa = mut[-1]
            pos = int(mut[1:-1])
            assert wt_aa == sequence[pos - 1],\
                    "%s: %s->%s (fasta WT: %s)"%(pos, wt_aa, mt_aa, sequence[pos - 1])
            sequence = sequence[:(pos - 1)] + mt_aa + sequence[pos:]
        return sequence

    def _mutations_to_sequences(self, mutations):
        return [self._mutation_to_sequence(m) for m in mutations]

    def _drop_invalid_mutation(self, df):
        '''
        Drop mutations WiM where
        - W is incosistent with the i-th AA in native_sequence
        - M is ambiguous, e.g., 'X'
        '''
        flags = []
        for mutation in df['mutant'].values:
            for mut in mutation.split(';'):
                wt_aa = mut[0]
                mt_aa = mut[-1]
                pos = int(mut[1:-1])
                valid = True if wt_aa == self.native_sequence[pos - 1] else False
                valid = valid and (mt_aa not in ['X'])
            flags.append(valid)
        df = df[flags].reset_index(drop=True)
        return df

    def _read_mutation_df(self, tsv):
        df = pd.read_table(tsv)
        df = self._drop_invalid_mutation(df)
        df['sequence'] = self._mutations_to_sequences(df['mutant'].values)
        return df

    def encode_glob_feat(self, sequences):
        feat = self.anhk_encoder.encode(sequences)
        feat = torch.from_numpy(feat).float()
        return feat

    def build_data(self, df, return_df=False):
        sequences = df['sequence'].values
        glob_feat = self.encode_glob_feat(sequences)
        
        labels = df['score'].values
        labels = torch.from_numpy(labels.astype(np.float32))

        samples = []
        for i in range(len(df)):
            sample = {
                'sequence':sequences[i],
                'label':labels[i],
            }
            sample['glob_feat'] = glob_feat[i]
            samples.append(sample)
        data = MetagenesisData(samples)
        if return_df:
            return data, df
        else:
            return data
        
    def get_loocv_dataloaders(self, fold_idx, batch_size=None):
        """
        Given a fold index, return training and test DataLoaders using leave-one-out.
        If batch_size is not provided, use the entire training fold in one batch.
        """
        # Use the combined training set; if available, use train_valid_df, else full_df.
        df = self.full_df.copy().reset_index(drop=True)
        n_samples = len(df)
        if fold_idx < 0 or fold_idx >= n_samples:
            raise ValueError(f"fold_idx must be between 0 and {n_samples-1}")

        # LOOCV: one sample is held out
        train_df = df.drop(fold_idx).reset_index(drop=True)
        test_df = df.iloc[[fold_idx]].reset_index(drop=True)

        train_data, _ = self.build_data(train_df, return_df=True)
        test_data, _ = self.build_data(test_df, return_df=True)

        # For low-N training you might set batch_size equal to the entire training fold
        if batch_size is None:
            batch_size = len(train_df)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1)
        return train_loader, train_df, test_loader, test_df

    def get_test_dataloader(self, batch_size=128,
            return_df=False):
        test_df = self.test_df.copy().reset_index(drop=True)
        test_data, test_df = self.build_data(test_df, return_df=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
        if return_df:
            return test_loader, test_df
        else:
            return test_loader

    def return_full_df(self):
        return self.full_df

if __name__ == '__main__':
    dataset = Dataset(
        test_tsv=f'../../TNet/A_c1/score/Mek1_AZD.tsv',
        fasta=f'../../TNet/A_c1/data/mek1/Mek1.fasta',
    )
    # dataset.build_data('train')
    # loader, df, _, _ = dataset.get_loocv_dataloaders(fold_idx=1)
    # print(df.head())
    # print(len(loader.__iter__()))
    loader, df = dataset.get_test_dataloader(batch_size=32, return_df=True)
    print(next(loader.__iter__()))
