import argparse
from seqeval.metrics import classification_report

parser = argparse.ArgumentParser()
parser.add_argument('pred_file')
parser.add_argument('--label_file','-l')
args = parser.parse_args()

labels = open(args.label_file,'r',encoding='utf-8').read().strip().split("\n")
labels  = [label.split(" ") for label in labels]

preds = open(args.pred_file,'r',encoding='utf-8').read().strip().split("\n")
preds = [pred.split(" ") for pred in preds]

print(classification_report(labels,preds))
