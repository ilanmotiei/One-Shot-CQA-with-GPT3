
from dataset import CQADataset
from official_evaluation_script import calculate_metrics
from model import CQAer
import torch
from torch.utils.data import DataLoader
import argparse
from utils import token_ids_to_sentence
import json
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-checkpoint', dest='model_checkpoint')
    parser.add_argument('--bert-max-input-len', dest='bert_max_input_len', default=512, type=int)
    parser.add_argument('--max-challenge-len', dest='max_challenge_len', default=10, type=int)
    parser.add_argument('--bert-model', dest='bert_model', default='bert-base-uncased')
    parser.add_argument('--bert-encoding-dim', dest='bert_encoding_dim', default=768, type=int)
    parser.add_argument('--history-window-size', dest='history_window_size', default=3, type=int)
    parser.add_argument('--device', dest='device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--include-in-span-layer', dest='include_in_span_layer', default=False, type=bool)
    parser.add_argument('--fine-tune-bert', dest='fine_tune_bert', default=False, type=bool)
    parser.add_argument('--batch-size', dest='batch_size', default=1, type=int)
    parser.add_argument('--num-workers', dest='num_workers', default=2, type=int)

    args = parser.parse_args()
    args.device = torch.device(args.device)

    return args


if __name__ == "__main__":
    OPTS = parse_args()
    model = CQAer(OPTS).to(OPTS.device)
    model.load_state_dict(torch.load(OPTS.model_checkpoint, map_location=OPTS.device))
    tokenizer = model.bert_tokenizer

    dev_dataset = CQADataset(coqa_dataset_filepath='coqa-dev-v1.0.json',
                             tokenizer=tokenizer,
                             max_challenges=-1,
                             max_challenge_len=OPTS.max_challenge_len,
                             args=OPTS)

    dev_dataloader = DataLoader(dataset=dev_dataset,
                                batch_size=OPTS.batch_size,
                                shuffle=False,
                                num_workers=OPTS.num_workers,
                                collate_fn=CQADataset.collate_fn)

    model.eval()
    predictions = []

    with torch.set_grad_enabled(False):
        for batch_idx, (batch_data, lengths, ids, turn_ids) in tqdm(enumerate(dev_dataloader)):
            batch_data = [t.to(OPTS.device) for t in batch_data]

            p_s_logits, p_e_logits, p_answer_logits, _, predicted_answers_ids = model(batch_data,
                                                                                      use_gold_answers=True,
                                                                                      return_predicted_answers=True)

            # p_s_logits.shape = (N, L, model.bert_max_input_len)
            # p_e_logits.shape = (N, L, model.bert_max_input_len)
            # p_answer_logits.shape = (N, L, model.bert_max_input_len, 4)

            N = p_s_logits.size(0)

            for challenge_index, challenge_id in zip(range(N), ids):
                for turn_index, turn_id in zip(range(int(lengths[challenge_index])), turn_ids[challenge_index]):
                    answer = token_ids_to_sentence(tokenizer, predicted_answers_ids[(challenge_index, turn_index)])
                    predictions.append({'id': challenge_id,
                                        'turn_id': turn_id,
                                        'answer': answer})

    json.dump(predictions, fp=open('predictions.json', 'w'), indent=2)

    metrics = calculate_metrics(data_filepath='coqa-dev-v1.0.json',
                                predictions_filepath='predictions.json')

    print("Results: --------------- ")
    print(metrics)
