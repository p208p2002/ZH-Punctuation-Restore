from core import BertTC
import torch
from torch.utils.data import Dataset,DataLoader
from core import get_tokenizer,make_model_config
from pytorch_lightning import Trainer

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
            print(window_tokens)
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
    
def merge_stride(output:int,window_size:int,step:int):
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

if __name__ == "__main__":
    window_size = 100
    step = 75

    text = "維基百科是維基媒體基金會運營的一個多語言的線上百科全書並以建立和維護作為開放式協同合作專案特點是自由內容自由編輯自由著作權目前是全球網路上最大且最受大眾歡迎的參考工具書名列全球二十大最受歡迎的網站其在搜尋引擎中排名亦較為靠前維基百科目前由非營利組織維基媒體基金會負責營運"
    dataset = DocumentDataset(text,window_size=window_size,step=step)
    dataloader = DataLoader(dataset=dataset,shuffle=False,batch_size=1)

    model = BertTC.load_from_checkpoint('/user_data/zhp/lightning_logs/version_0/checkpoints/last.ckpt')
    trainer =Trainer()
    model_pred_out = trainer.predict(model,dataloader)
    merge_pred_result = merge_stride(model_pred_out,window_size,step)
    merge_pred_result_deocde = decode_pred(merge_pred_result)
    merge_pred_result_deocde = ''.join(merge_pred_result_deocde)
    print(merge_pred_result_deocde)

    


