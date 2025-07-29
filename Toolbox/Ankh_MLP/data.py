import warnings
import numpy as np
import pandas as pd
from Bio import SeqIO
import torch.utils.data
from sklearn.model_selection import KFold, ShuffleSplit

from global_feature import AnhkEncoder


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
            fasta=None,
            split_ratio=[0.875, 0.125],
            random_seed=42):
        """
        split_ratio: [train, valid] or [train, valid, test]
        """

        self.train_tsv = train_tsv
        self.test_tsv = test_tsv
        self.fasta = fasta
        
        self.split_ratio = split_ratio
        self.rng = np.random.RandomState(random_seed)

        self.native_sequence = self._read_native_sequence()
        if train_tsv is not None:
            self.full_df = self._read_mutation_df(train_tsv)
        else:
            self.full_df = None

        if test_tsv is None:
            if len(split_ratio) != 3:
                split_ratio = [0.7, 0.1, 0.2]
                warnings.warn("\nsplit_ratio should have 3 elements if test_tsv is None." + \
                    f"Changing split_ratio to {split_ratio}. " + \
                    "Set to other values using --split_ratio.")
            self.train_df, self.valid_df, self.test_df = \
                self._split_dataset_df(self.full_df, split_ratio)
        else:
            if len(split_ratio) != 2:
                split_ratio = [0.875, 0.125]
                warnings.warn("\nsplit_ratio should have 2 elements if test_tsv is provided. " + \
                    f"Changing split_ratio to {split_ratio}. " + \
                    "Set to other values using --split_ratio.")
            self.test_df = self._read_mutation_df(test_tsv)
            if self.full_df is not None:
                self.train_df, self.valid_df, _ = \
                    self._split_dataset_df(self.full_df, split_ratio)

        if self.full_df is not None:
            self.train_valid_df = pd.concat(
                [self.train_df, self.valid_df]).reset_index(drop=True)

        self.anhk_encoder = AnhkEncoder()

    def _read_native_sequence(self):
        fasta = SeqIO.read(self.fasta, 'fasta')
        native_sequence = str(fasta.seq)
        return native_sequence

    def _check_split_ratio(self, split_ratio):
        """
        Modified from: https://github.com/pytorch/text/blob/3d28b1b7c1fb2ddac4adc771207318b0a0f4e4f9/torchtext/data/dataset.py#L284-L311
        """
        test_ratio = 0.
        if isinstance(split_ratio, float):
            assert 0. < split_ratio < 1., (
                "Split ratio {} not between 0 and 1".format(split_ratio))
            valid_ratio = 1. - split_ratio
            return (split_ratio, valid_ratio, test_ratio)
        elif isinstance(split_ratio, list):
            length = len(split_ratio)
            assert length == 2 or length == 3, (
                "Length of split ratio list should be 2 or 3, got {}".format(split_ratio))
            ratio_sum = sum(split_ratio)
            if not ratio_sum == 1.:
                split_ratio = [float(ratio) / ratio_sum for ratio in split_ratio]
            if length == 2:
                return tuple(split_ratio + [test_ratio])
            return tuple(split_ratio)
        else:
            raise ValueError('Split ratio must be float or a list, got {}'
                            .format(type(split_ratio)))

    def _split_dataset_df(self, input_df, split_ratio, resample_split=False):
        """
        Modified from:
        https://github.com/pytorch/text/blob/3d28b1b7c1fb2ddac4adc771207318b0a0f4e4f9/torchtext/data/dataset.py#L86-L136
        """
        _rng = self.rng.randint(512) if resample_split else self.rng
        df = input_df.copy()
        df = df.sample(frac=1, random_state=_rng).reset_index(drop=True)
        N = len(df)
        train_ratio, valid_ratio, test_ratio = self._check_split_ratio(split_ratio)
        train_len = int(round(train_ratio * N))
        valid_len = N - train_len if not test_ratio else int(round(valid_ratio * N))

        train_df = df.iloc[:train_len].reset_index(drop=True)
        valid_df = df.iloc[train_len:train_len + valid_len].reset_index(drop=True)
        test_df = df.iloc[train_len + valid_len:].reset_index(drop=True)

        return train_df, valid_df, test_df

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

    def build_data(self, mode, return_df=False):
        if mode == 'train':
            df = self.train_df.copy()
        elif mode == 'valid':
            df = self.valid_df.copy()
        elif mode == 'test':
            df = self.test_df.copy()
        else:
            raise NotImplementedError

        sequences = df['sequence'].values
        print(sequences.size)
        glob_feat = self.encode_glob_feat(sequences)
        print(glob_feat.shape)
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

    def get_dataloader(self, mode, batch_size=128,
            return_df=False, resample_train_valid=False):
        if resample_train_valid:
            self.train_df, self.valid_df, _ = \
                self._split_dataset_df(
                    self.train_valid_df, self.split_ratio[:2], resample_split=True)

        if mode == 'train_valid':
            train_data, train_df = self.build_data('train', return_df=True)
            valid_data, valid_df = self.build_data('valid', return_df=True)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
            valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
            if return_df:
                return (train_loader, train_df), (valid_loader, valid_df)
            else:
                return train_loader, valid_loader
        elif mode == 'test':
            test_data, test_df = self.build_data('test', return_df=True)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
            if return_df:
                return test_loader, test_df
            else:
                return test_loader
        else:
            raise NotImplementedError

if __name__ == '__main__':
    dataset = Dataset(
        train_tsv='../esm-MLP/data/scale_tt.tsv',
        fasta='../../TNet/data/kras17N.fasta',
        split_ratio=[0.7, 0.1, 0.2],
    )
    # dataset.build_data('train')
    '''
    (loader, df), (_, _) = dataset.get_dataloader('train_valid',
        batch_size=32, return_df=True)
    print(df.head())
    print(len(loader.__iter__()))
    (loader, df), (_, _) = dataset.get_dataloader('train_valid',
        batch_size=32, return_df=True, resample_train_valid=True)
    print(df.head())
    print(len(loader.__iter__()))
    '''
    loader, df = dataset.get_dataloader('test',
        batch_size=32, return_df=True, resample_train_valid=True)
    #print(next(loader.__iter__()))
