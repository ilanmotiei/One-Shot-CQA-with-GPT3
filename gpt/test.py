
import torch
import configurations as cfg
import sys
sys.path.append('..')
from cqa.model import CQAer

from transformers import BertModel, BertTokenizer


if __name__ == "__main__":

    device = torch.device('cuda')
    cqaer = CQAer(torch.load(open(cfg.cqaer_args_filepath, 'rb')))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_checkpoint = cqaer.bert.to(device)
    index = torch.load(cfg.index_filepath)
    index['challenges_embeddings'] = index['challenges_embeddings'].to(device)