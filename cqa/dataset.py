
from torch.utils.data import Dataset
from cqa.utils import Challenge
import json
from collections import defaultdict
from transformers import BertTokenizer
from torch.nn.functional import pad
import torch
from tqdm import tqdm


class CQADataset(Dataset):

    def __init__(self, coqa_dataset_filepath, tokenizer, max_challenges, max_challenge_len, args):
        super().__init__()

        self.bert_max_input_len = args.bert_max_input_len

        data = json.load(open(coqa_dataset_filepath, 'r'))['data']

        splitted_data = []

        for challenge in data:
            splitted_data += CQADataset.split_challenge(challenge, max_challenge_len)

        self.challenges_list = []

        for i in tqdm(range(len(splitted_data))):
            challenge = splitted_data[i]
            preprocessed_challenge = Challenge(challenge=challenge,
                                               tokenizer=tokenizer,
                                               bert_max_input_len=self.bert_max_input_len,
                                               preprocessed=(not args.preprocess_data))
            if preprocessed_challenge.discard:
                continue
            # else:

            self.challenges_list.append(preprocessed_challenge)

            if len(self.challenges_list) == max_challenges:
                break

        if args.preprocess_data:
            # save the preprocessed data
            json.dump({'data': data}, open(f"{'.'.join(coqa_dataset_filepath.split('.')[:-1])}_preprocessed.json", 'w'),
                      indent=2)

    def __len__(self):
        return len(self.challenges_list)

    def __getitem__(self, idx):
        challenge = self.challenges_list[idx]

        return (challenge.answer_span_start.long(), challenge.answer_span_end.long(),
                challenge.question_n_context_ids.long(), challenge.question_n_context_mask.long(), challenge.question_n_context_token_types.long(),
                challenge.question_n_context_context_start_idx.int(), challenge.question_n_context_context_end_idx.int(),
                challenge.answer_n_context_ids.long(), challenge.answer_n_context_mask.long(), challenge.answer_n_context_token_types.long(),
                challenge.answer_n_context_context_start_idx.long(), challenge.answer_n_context_context_end_idx.long(),
                challenge.category.long()), challenge.id, challenge.turn_ids

    @staticmethod
    def collate_fn(challenges):
        """
        :param challenges: A list of N challenges.
                           The variable 'sequences' bellow contains a list of N tuples of tensors.
        :return: * The tensors, after batchified.
                 * The lengths of the tensors before batchified.

                 The tensors of shape (L_i, bert_max_input_size) become of shpae (max_i(L_i), bert_max_input_size).
                 The tensors of shape (L_i) become of shape (max_i(L_i)).

                 Batchifing is done by padding with -1 's.
        """

        sequences, ids, turn_ids = list(zip(*challenges))

        max_li = max([sequence[0].size(0) for sequence in sequences])
        batch = []

        for ts in zip(*sequences):
            if len(ts[0].shape) == 2:
                batchified_ts = torch.stack([torch.cat([t, torch.zeros(size=(max_li-t.size(0), t.size(1)))]) for t in ts])
            if len(ts[0].shape) == 1:
                batchified_ts = torch.stack([torch.cat([t, -torch.ones(size=(max_li-t.size(0), ))]) for t in ts])

            batch += [batchified_ts.long()]

        return batch, torch.Tensor([sequence[0].size(0) for sequence in sequences]), ids, turn_ids

    @staticmethod
    def split_challenge(challenge, max_sub_challenge_len):
        sub_challenges = []
        total_challenge_len = len(challenge['questions'])

        if max_sub_challenge_len >= total_challenge_len or max_sub_challenge_len == -1:
            return [challenge]

        for i in range(total_challenge_len // max_sub_challenge_len + (total_challenge_len % max_sub_challenge_len != 0)):
            sub_challenges.append({'id': challenge['id'],
                                   'story': challenge['story'],
                                   'questions': challenge['questions'][max_sub_challenge_len * i: max_sub_challenge_len * (i+1)],
                                   'answers': challenge['answers'][max_sub_challenge_len * i: max_sub_challenge_len * (i+1)]})

        return sub_challenges


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    CQADataset(coqa_dataset_filepath='coqa-train-v1.0.json',
               tokenizer=tokenizer,
               bert_max_input_len=512)
