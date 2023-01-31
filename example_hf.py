from zhpr.predict import DocumentDataset,merge_stride,decode_pred
from transformers import AutoModelForTokenClassification,AutoTokenizer
from torch.utils.data import DataLoader

def predict_step(batch,model,tokenizer):
        assert batch.shape[0]==1
        out = []
        input_ids = batch
        encodings = {'input_ids': input_ids}
        output = model(**encodings)

        predicted_token_class_id_batch = output['logits'].argmax(-1)
        for predicted_token_class_ids, input_ids in zip(predicted_token_class_id_batch, input_ids):
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            
            # compute the pad start in input_ids
            # and also truncate the predict
            input_ids = input_ids.tolist()
            try:
                input_id_pad_start = input_ids.index(tokenizer.pad_token_id)
            except:
                input_id_pad_start = len(input_ids)
            input_ids = input_ids[:input_id_pad_start]
            tokens = tokens[:input_id_pad_start]
    
            # predicted_token_class_ids
            predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids]
            predicted_tokens_classes = predicted_tokens_classes[:input_id_pad_start]

            for token,ner in zip(tokens,predicted_tokens_classes):
                out.append((token,ner))
        return out

if __name__ == "__main__":
    window_size = 100
    step = 75
    text = "維基百科是維基媒體基金會運營的一個多語言的線上百科全書並以建立和維護作為開放式協同合作專案特點是自由內容自由編輯自由著作權目前是全球網路上最大且最受大眾歡迎的參考工具書名列全球二十大最受歡迎的網站其在搜尋引擎中排名亦較為靠前維基百科目前由非營利組織維基媒體基金會負責營運"
    dataset = DocumentDataset(text,window_size=window_size,step=step)
    dataloader = DataLoader(dataset=dataset,shuffle=False,batch_size=1)

    model_name = 'p208p2002/zh-wiki-punctuation-restore'
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_pred_out = []
    for batch in dataloader:
        model_pred_out.append(predict_step(batch,model,tokenizer))
        
    merge_pred_result = merge_stride(model_pred_out,step)
    merge_pred_result_deocde = decode_pred(merge_pred_result)
    merge_pred_result_deocde = ''.join(merge_pred_result_deocde)
    print(merge_pred_result_deocde)

    
