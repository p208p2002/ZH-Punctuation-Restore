import pandas as pd
from sklearn.utils import shuffle
from torch.utils.data import Dataset, random_split
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from transformers import BertTokenizerFast, BertForNextSentencePrediction
import torch
from torch import optim
import pytorch_lightning as pl


def get_tokenizer():
    return BertTokenizerFast.from_pretrained('bert-base-chinese')


def get_model():
    return BertForNextSentencePrediction.from_pretrained('bert-base-chinese')


class STCDataset(Dataset):
    def __init__(self, file_path) -> None:
        super().__init__()
        self.tokenizer = get_tokenizer()

        df = pd.read_excel(file_path)
        df_pos = df.loc[df["sum"] >= 0]
        df_neg = df.loc[df["sum"] < 0]
        balance_num = min(len(df_pos.index), len(df_neg.index))

        df_pos = shuffle(df_pos)[:balance_num]
        df_pos['label'] = [0]*balance_num

        df_neg = shuffle(df_neg)[:balance_num]
        df_neg['label'] = [1]*balance_num

        df_bal = pd.concat([df_pos, df_neg])
        self.df = df_bal

    def __getitem__(self, index):
        data = self.df.iloc[index]
        encodings = self.tokenizer(
            data["query"],
            data["comment"],
            max_length=120,
            return_tensors="pt",
            truncation='only_second',
            padding='max_length'

        )
        return (
            encodings['input_ids'][0],
            encodings['attention_mask'][0],
            encodings['token_type_ids'][0],
            torch.tensor(data["label"])
        )

    def __len__(self):
        return len(self.df.index)


class STCDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=8):
        super().__init__()
        self.batch_size = batch_size
        dataset = STCDataset("data/STC2_Train_V1.0.xlsx")

        train_set_size = int(len(dataset) * 0.8)
        valid_set_size = int(len(dataset) * 0.1)
        test_set_size = len(dataset) - (train_set_size + valid_set_size)
        train_set, valid_set, test_set = random_split(
            dataset, [train_set_size, valid_set_size, test_set_size])

        self.train = train_set
        self.valid = valid_set
        self.test = test_set

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


class BertNSP(pl.LightningModule):
    # 0 indicates sequence B is a continuation of sequence A,
    # 1 indicates sequence B is a random sequence.

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(self.args)

        self.model = get_model()
        self.tokeizer = get_tokenizer()

    def training_step(self, batch, batch_idx):
        encodings = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2],
            'labels': batch[3]
        }
        output = self.model(**encodings)
        loss = output['loss']

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        encodings = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2],
            'labels': batch[3]
        }
        output = self.model(**encodings)
        loss = output['loss']

        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        encodings = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2],
            'labels': batch[3]
        }
        output = self.model(**encodings)
        loss = output['loss']

        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return optimizer
