from zhpr.predict import DocumentDataset,merge_stride,decode_pred
from zhpr.core import ZhprBert
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

if __name__ == "__main__":
    window_size = 100
    step = 75
    text = "維基百科是維基媒體基金會運營的一個多語言的線上百科全書並以建立和維護作為開放式協同合作專案特點是自由內容自由編輯自由著作權目前是全球網路上最大且最受大眾歡迎的參考工具書名列全球二十大最受歡迎的網站其在搜尋引擎中排名亦較為靠前維基百科目前由非營利組織維基媒體基金會負責營運"
    dataset = DocumentDataset(text,window_size=window_size,step=step)
    dataloader = DataLoader(dataset=dataset,shuffle=False,batch_size=1)

    model = ZhprBert.load_from_checkpoint('/user_data/zhp/lightning_logs/version_1/checkpoints/val-epoch=epoch=01-setp=step=5987-val_loss=val_loss=0.09140.ckpt')
    trainer =Trainer()
    model_pred_out = trainer.predict(model,dataloader)
    merge_pred_result = merge_stride(model_pred_out,step)
    merge_pred_result_deocde = decode_pred(merge_pred_result)
    merge_pred_result_deocde = ''.join(merge_pred_result_deocde)
    print(merge_pred_result_deocde)

    
