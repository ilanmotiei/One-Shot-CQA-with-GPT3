"""
Creates an embedding vector (of shape (768, )) for each challenge.
That's for 1-shot GPT-3 usage. For an example at the dev set, we give the model the closest example of challenge,
with respect for the current challenge we're looking at in the dev set, + the current prefix of the challenge.
"""

import sys
sys.path.append('..')
from cqa.model import CQAer

from transformers import BertTokenizer, BertModel
import torch
from typing import List, Union
import json
from tqdm import tqdm

import configurations as cfg


def encode(tokenizer: BertTokenizer, passage: str, question: str, device: torch.device):
    encoding = tokenizer.encode_plus(passage, question)
    input_ids = encoding['input_ids']
    token_type_ids = encoding['token_type_ids']
    attention_masks = encoding['attention_mask']

    if len(input_ids) > cfg.max_input_len:
        raise IndexError

    input_ids += [0] * (cfg.max_input_len - len(input_ids))
    token_type_ids += [0] * (cfg.max_input_len - len(token_type_ids))
    attention_masks += [0] * (cfg.max_input_len - len(attention_masks))

    input_ids = torch.Tensor(input_ids).int().to(device)
    token_type_ids = torch.Tensor(token_type_ids).int().to(device)
    attention_masks = torch.Tensor(attention_masks).int().to(device)

    return input_ids, token_type_ids, attention_masks


def create_embedding(bert_checkpoint: BertModel,
                     tokenizer: BertTokenizer,
                     passage: str,
                     questions: List[str],
                     device: torch.device,
                     num_questions_for_embedding: int,
                     from_dev: bool,  # indicates whether it's a sample from the dev set or from the train set
                     j=0  # The index of the current question we're looking at
                     ) -> Union[torch.Tensor, None]:

    total_input_ids = []
    total_token_type_ids = []
    total_attention_masks = []
    challenge_embedding = 0
    valid = 0

    if num_questions_for_embedding > 0:
        if from_dev:
            # we take the last questions
            questions = questions[max(0, j-num_questions_for_embedding+1): j+1]
        else:
            # we take the first questions
            questions = questions[:num_questions_for_embedding]

    if num_questions_for_embedding > 0:
        for i, question in enumerate(questions):
            try:
                input_ids, token_type_ids, attention_masks = encode(tokenizer, passage, question, device)
            except IndexError:
                # len(input_ids) > max_input_len
                continue

            total_input_ids.append(input_ids)
            total_token_type_ids.append(token_type_ids)
            total_attention_masks.append(attention_masks)

            if (i + 1) % cfg.batch_size == 0 or i == (len(questions) - 1):
                input_ids = torch.stack(total_input_ids)
                token_type_ids = torch.stack(total_token_type_ids)
                attention_masks = torch.stack(total_attention_masks)

                with torch.no_grad():
                    _, cls = bert_checkpoint(input_ids, attention_masks, token_type_ids, return_dict=False)

                # cls.shape = (batch_size, 768)

                challenge_embedding += torch.sum(cls, dim=0)

                total_input_ids = []
                total_token_type_ids = []
                total_attention_masks = []

                torch.cuda.empty_cache()

            valid += 1
    else:
        try:
            input_ids, token_type_ids, attention_masks = encode(tokenizer, passage, '', device)
            input_ids = input_ids.unsqueeze(0)
            token_type_ids = token_type_ids.unsqueeze(0)
            attention_masks = attention_masks.unsqueeze(0)

            with torch.no_grad():
                _, cls = bert_checkpoint(input_ids, attention_masks, token_type_ids, return_dict=False)
                # ^ : shape = (1, 768)
                challenge_embedding = cls.squeeze(0)
                valid = 1
        except IndexError:
            # The challenge's passage is longer than cfg.max_input_len
            return None

    if isinstance(challenge_embedding, int):
        # all the passage + question for every question in the current challenge where of length > max_input_len:
        return None

    challenge_embedding /= valid

    return challenge_embedding


def create_train_index(bert_checkpoint: BertModel,
                       bert_tokenizer: BertTokenizer,
                       train_data_filepath: str,
                       index_filepath: str,
                       num_questions_in_embedding: int,
                       device: torch.device) -> None:
    """
    :param bert_checkpoint: A checkpoint of a bert model you'd like to use for getting the challenges' embeddings.
    :param bert_tokenizer: A tokenizer corresponding to the bert checkpoint you've sent to the function.
    :param train_data_filepath: The path to the file containing the training data of the CO-QA dataset.
    :param index_filepath: The .pth file in which we'll store the embedding tensor for every of the ~8000 challenges
                            at the CO-QA dataset.
    :param num_questions_in_embedding: Only the first <num_questions_in_embedding> questions in each challenge will
                                        be considered at the challenge's embedding.
    :param device: The device we're using for creating the embeddings.
    """

    train_data_dict = json.load(open(train_data_filepath, 'rb'))['data']
    challenges_embeddings = []
    idx_to_challenge_idx = {}

    curr_embedding_index = 0
    for challenge_idx, challenge in tqdm(enumerate(train_data_dict)):
        challenge_embedding = create_embedding(bert_checkpoint=bert_checkpoint,
                                               tokenizer=bert_tokenizer,
                                               passage=challenge['story'],
                                               questions=[q['input_text'] for q in challenge['questions']],
                                               num_questions_for_embedding=num_questions_in_embedding,
                                               device=device,
                                               from_dev=False)

        if challenge_embedding is None:
            continue

        challenges_embeddings.append(challenge_embedding)
        idx_to_challenge_idx[curr_embedding_index] = challenge_idx
        curr_embedding_index += 1

    challenges_embeddings = torch.stack(challenges_embeddings)  # shape = (#valid_challenges, 768)

    index = {'challenges_embeddings': challenges_embeddings,
             'idx_to_challenge_idx': idx_to_challenge_idx}

    torch.save(index, index_filepath)


def find_closest_challenge(bert_checkpoint: BertModel,
                           tokenizer: BertTokenizer,
                           index,
                           train_data,
                           devset_challenge,
                           num_questions_in_embedding,
                           device,
                           j  # The index of the question we're currently at in the given challenge
                           ) -> Union[dict, None]:

    devset_challenge_embeddings = create_embedding(bert_checkpoint=bert_checkpoint,
                                                   tokenizer=tokenizer,
                                                   passage=devset_challenge['story'],
                                                   questions=[q['input_text'] for q in devset_challenge['questions']],
                                                   num_questions_for_embedding=num_questions_in_embedding,
                                                   device=device,
                                                   from_dev=True,
                                                   j=j)

    devset_challenge_embeddings = devset_challenge_embeddings.to(device).unsqueeze(dim=1)  # shape = (768, 1)

    embedding_index = torch.argmax(torch.matmul(index['challenges_embeddings'], devset_challenge_embeddings), dim=0).item()
    challenge_index = index['idx_to_challenge_idx'][embedding_index]

    return train_data[challenge_index]


if __name__ == "__main__":
    device = torch.device('cuda')
    cqaer = CQAer(torch.load(open(cfg.cqaer_args_filepath, 'rb')))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_checkpoint = cqaer.bert.to(device)

    create_train_index(bert_checkpoint=cqaer.bert.to(device),
                       bert_tokenizer=tokenizer,
                       train_data_filepath=cfg.train_data_filepath,
                       index_filepath=cfg.index_filepath,
                       num_questions_in_embedding=cfg.questions_in_embedding,
                       device=device)

    # index = torch.load('index_for_gpt.pth')
    # index['challenges_embeddings'] = index['challenges_embeddings'].to(device)
    # train_data = json.load(open('./coqa-train-v1.0.json', 'rb'))['data']
    # dev_data = json.load(open('./coqa-dev-v1.0.json', 'rb'))['data']

    # curr_devset_challenge = dev_data[100]
    # closest_trainset_challenge = find_closest_challenge(bert_checkpoint=bert_checkpoint,
    #                                                     tokenizer=tokenizer,
    #                                                     index=index,
    #                                                     train_data=train_data,
    #                                                     devset_challenge=curr_devset_challenge,
    #                                                     device=device)
    #
    # print("----------------")
    # print(curr_devset_challenge)
    # print("----------------")
    # print(closest_trainset_challenge)
