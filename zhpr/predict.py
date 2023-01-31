import torch
from torch.utils.data import Dataset
from .core import get_tokenizer,make_model_config

class DocumentDataset(Dataset):
    def __init__(self, document:str, window_size=384,step=307) -> None:
        super().__init__()
        self.tokenizer = get_tokenizer()
        self.config = make_model_config()
        self.document = document
        self.window_size = window_size
        self.step = step
        self.data = list(self._stride(self.window_size))
        
    def _stride(self, window_size):
        
        tokens = list(self.document)
        for window_start in range(0, len(tokens), self.step):
            window_tokens = tokens[window_start:window_start+window_size]
            yield {
                'tokens': window_tokens,
            }

    def __getitem__(self, index):
        data = self.data[index]
        tokens = self.tokenizer.convert_tokens_to_ids(data['tokens'])
        while len(tokens) < self.window_size:
            tokens.append(self.tokenizer.pad_token_id)

        return torch.tensor(tokens)

    def __len__(self):
        return len(self.data)
    
def merge_stride(output:int,step:int):
    out = []
    for sent_idx,stride_sent in enumerate(output):
        token_idx = step*sent_idx
        for token_ner in stride_sent:
            if token_idx + 1 > len(out):
                out.append(token_ner)
            else:
                out[token_idx] = token_ner
            token_idx += 1
    return out
    
def decode_pred(token_ners):
    out = []
    for token_ner in token_ners:
        out.append(token_ner[0])
        if token_ner[-1] != 'O':
            out.append(token_ner[-1][-1])
    return out


