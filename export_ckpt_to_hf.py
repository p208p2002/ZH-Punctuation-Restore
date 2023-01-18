import argparse
from core import BertTC
import os
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out','-o',help='path to save')
    parser.add_argument('ckpt_path')
    args = parser.parse_args()

    ModelClass = BertTC
    
    assert os.path.isfile(args.ckpt_path)
    lighting_model = ModelClass.load_from_checkpoint(args.ckpt_path)
    lighting_model.model.save_pretrained(args.out)
    lighting_model.tokenizer.save_pretrained(args.out)
    print("Done")
