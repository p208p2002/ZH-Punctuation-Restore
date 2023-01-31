
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import BertTokenizerFast, BertForTokenClassification, BertConfig
import torch
from torch import optim
import pytorch_lightning as pl
import json
import os

USE_ZH_PUNCTUATION = [
    '，',
    '、',
    '。',
    '？',
    '！',
    '；',
]


def make_model_config():
    label2id = {}
    id2label = {}

    for x_id, x in enumerate(['O'] + USE_ZH_PUNCTUATION):
        label = f"S-{x}"
        if x_id == 0:
            label = "O"
        label2id[label] = x_id
        id2label[x_id] = label

    return BertConfig.from_pretrained(
        'bert-base-chinese',
        label2id=label2id,
        id2label=id2label,
        num_labels=len(USE_ZH_PUNCTUATION) + 1
    )

def get_tokenizer():
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    return tokenizer


def get_model():
    return BertForTokenClassification.from_pretrained('bert-base-chinese', config=make_model_config(), ignore_mismatched_sizes=True)


class _PunctDataset(Dataset):
    def __init__(self, file_path) -> None:
        super().__init__()
        self.data = open(
            file_path, "r", encoding='utf-8').read().strip().split("\n")

    def __getitem__(self, index):
        data = self.data[index]
        return json.loads(data)

    def __len__(self):
        return len(self.data)


class PunctDataset(Dataset):
    def __init__(self, file_path, window_size=384) -> None:
        super().__init__()
        self.tokenizer = get_tokenizer()
        self.config = make_model_config()
        self.dataset = _PunctDataset(file_path)
        self.window_size = window_size
        self.data = list(self._stride(self.window_size))

    def _stride(self, window_size):
        step = int(window_size*0.8)
        for data_idx in range(len(self.dataset)):
            data = self.dataset[data_idx]
            tokens = data['tokens']
            bios = data['bios']
            for window_start in range(0, len(tokens), step):
                window_tokens = tokens[window_start:window_start+window_size]
                window_bios = bios[window_start:window_start+window_size]
                yield {
                    'tokens': window_tokens,
                    'bios': window_bios
                }

    def __getitem__(self, index):
        data = self.data[index]
        tokens = self.tokenizer.convert_tokens_to_ids(data['tokens'])
        bios = [self.config.label2id[x] for x in data['bios']]
        while len(tokens) < self.window_size:
            tokens.append(self.tokenizer.pad_token_id)
            bios.append(-100)

        return torch.tensor(tokens), torch.tensor(bios)

    def __len__(self):
        return len(self.data)


class PunctDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=8):
        super().__init__()
        self.batch_size = batch_size
        self.train = PunctDataset(
            'data/ZH-Wiki-Punctuation-Restore-Dataset/train.jsonl')
        self.valid = PunctDataset(
            'data/ZH-Wiki-Punctuation-Restore-Dataset/dev.jsonl')
        self.test = PunctDataset(
            'data/ZH-Wiki-Punctuation-Restore-Dataset/test.jsonl')

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


class ZhprBert(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(self.args)

        self.model = get_model()
        self.tokenizer = get_tokenizer()

    def training_step(self, batch, batch_idx):
        encodings = {
            'input_ids': batch[0],
            'labels': batch[-1]
        }
        output = self.model(**encodings)
        loss = output['loss']
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        encodings = {
            'input_ids': batch[0],
            'labels': batch[-1]
        }
        output = self.model(**encodings)
        loss = output['loss']

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        encodings = {
            'input_ids': batch[0],
            'labels': batch[-1]
        }
        output = self.model(**encodings)
        loss = output['loss']

        tf = open(os.path.join(self.trainer.log_dir, '_test.label'), 'a')
        with open(os.path.join(self.trainer.log_dir, 'pred.log'), 'a') as f:
            predicted_token_class_id_batch = output['logits'].argmax(-1)
            for predicted_token_class_ids, labels in zip(predicted_token_class_id_batch, labels := batch[-1]):

                # compute the pad start in lable
                # and also truncate the predict
                labels = labels.tolist()
                try:
                    labels_pad_start = labels.index(-100)
                except:
                    labels_pad_start = len(labels)
                labels = labels[:labels_pad_start]
                
                # predicted_token_class_ids
                predicted_tokens_classes = [self.model.config.id2label[t.item()] for t in predicted_token_class_ids]
                predicted_tokens_classes = predicted_tokens_classes[:labels_pad_start]
                predicted_tokens_classe_pred = ' '.join(
                    predicted_tokens_classes)
                f.write(f"{predicted_tokens_classe_pred}\n")

                # labels
                labels_tokens_classes = [self.model.config.id2label[t] for t in labels]
                labels_tokens_classes = ' '.join(labels_tokens_classes)
                tf.write(f"{labels_tokens_classes}\n")

        self.log("test_loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        assert batch.shape[0]==1
        out = []
        input_ids = batch
        encodings = {'input_ids': input_ids}
        output = self.model(**encodings)

        predicted_token_class_id_batch = output['logits'].argmax(-1)
        for predicted_token_class_ids, input_ids in zip(predicted_token_class_id_batch, input_ids):
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            
            # compute the pad start in input_ids
            # and also truncate the predict
            input_ids = input_ids.tolist()
            try:
                input_id_pad_start = input_ids.index(self.tokenizer.pad_token_id)
            except:
                input_id_pad_start = len(input_ids)
            input_ids = input_ids[:input_id_pad_start]
            tokens = tokens[:input_id_pad_start]
    
            # predicted_token_class_ids
            predicted_tokens_classes = [self.model.config.id2label[t.item()] for t in predicted_token_class_ids]
            predicted_tokens_classes = predicted_tokens_classes[:input_id_pad_start]

            for token,ner in zip(tokens,predicted_tokens_classes):
                out.append((token,ner))
        return out
            

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args.learning_rate)
        return optimizer
