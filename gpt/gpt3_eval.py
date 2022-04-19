import sys
sys.path.append('..')
from cqa_1.model import CQAer

import torch
from transformers import BertTokenizer, BertModel
import json
import random
from create_embeddings import find_closest_challenge
from gpt3_api import GPT_Completion
from cqa_1.official_evaluation_script import calculate_metrics
import tqdm
import configurations as cfg
from create_embeddings import create_train_index
import time
from collections import defaultdict


def format_input(example_challenge: dict,
                 challenge: dict,
                 question_idx: int):
    """
    :param example_challenge: An in-context challenge to give to GPT-3.
    :param challenge: A new challenge from the dev-set to give to GPT-3.
    :param question_idx: The index of the question from the challenge at the 'dev' set to be queried.
    :return: A string specifying the appropriate formatted input to give to GPT-3.
    """

    formatted_gpt_input = []
    formatted_gpt_input.append("Find the span containing the answer for the following questions regarding the story")
    example_challenge_story = ' '.join(example_challenge['story'].split('\n'))
    formatted_gpt_input.append(f"story: {example_challenge_story}")

    for question, answer in zip(example_challenge['questions'], example_challenge['answers']):
        formatted_gpt_input.append(f"question: {question['input_text']}")
        formatted_gpt_input.append(f"answer: {answer['input_text']}")

    formatted_gpt_input.append("Find the span containing the answer for the following questions regarding the story")
    challenge_story = ' '.join(challenge['story'].split('\n'))
    formatted_gpt_input.append(f"story: {challenge_story}")

    for question, answer in zip(challenge['questions'][:question_idx], challenge['answers'][:question_idx]):
        formatted_gpt_input.append(f"question: {question['input_text']}")
        formatted_gpt_input.append(f"answer: {answer['input_text']}")

    formatted_gpt_input.append(f"question: {challenge['questions'][question_idx]['input_text']}")
    formatted_gpt_input.append('answer: ')

    formatted_gpt_input = '\n'.join(formatted_gpt_input)

    return formatted_gpt_input


def embeddings_test_bert(model, tokenizer, index, train_data, dev_data, device, num_questions_in_embedding, predictions_filepath, logs_file):
    predictions = []
    same_category = 0
    total = 0
    
    for c in tqdm.tqdm(dev_data):
        for j, answer in enumerate(c['answers']):
            closest_trainset_challenge = find_closest_challenge(bert_checkpoint=model,
                                                                tokenizer=tokenizer,
                                                                index=index,
                                                                train_data=train_data,
                                                                devset_challenge=c,
                                                                num_questions_in_embedding=num_questions_in_embedding,
                                                                device=device,
                                                                j=j)

            same_category += c['source'] == closest_trainset_challenge['source']
            total += 1

            gpt3_input = format_input(example_challenge=closest_trainset_challenge, challenge=c, question_idx=j)
            gpt3_output = GPT_Completion(gpt3_input)
            answer['input_text'] = gpt3_output
            predictions.append({'id': c['id'],
                                'turn_id': answer['turn_id'],
                                'answer': gpt3_output})

            time.sleep(2)
            
    json.dump(predictions, open(predictions_filepath, 'w'), indent=2)
            
    metrics = calculate_metrics(data_filepath=cfg.gpt_dev_subset_path,
                                predictions_filepath=predictions_filepath)
    
    print("Results: ")
    print(metrics)

    logs_file.write(f" ---------- k = {num_questions_in_embedding} ---------- ")
    logs_file.write(str(metrics))
    
    print(f"% Times that the context challenge was from the same source as the queried challenge: {same_category / total * 100}", file=logs_file)

    logs_file.flush()


def random_test(train_data, dev_data, predictions_filepath, logs_file, method='same_source'):
    predictions = []
    same_category = 0
    total = 0

    if method == 'same_source':
        source_to_challenges = defaultdict(lambda: [])
        for c in train_data:
            source_to_challenges[c['source']].append(c)

    for c in tqdm.tqdm(dev_data):
        if method == 'same_source':
            example_challenge = random.choice(source_to_challenges[c['source']])
        else:
            example_challenge = random.choice(train_data)

        same_category += c['source'] == example_challenge['source']
        total += 1

        for j, answer in enumerate(c['answers']):
            gpt3_input = format_input(example_challenge=example_challenge, challenge=c, question_idx=j)
            gpt3_output = GPT_Completion(gpt3_input)
            answer['input_text'] = gpt3_output
            predictions.append({'id': c['id'],
                                'turn_id': answer['turn_id'],
                                'answer': gpt3_output})

            time.sleep(3)

    json.dump(predictions, open(predictions_filepath, 'w'), indent=2)

    metrics = calculate_metrics(data_filepath=cfg.gpt_dev_subset_path,
                                predictions_filepath=predictions_filepath)

    print("Results: ")
    print(metrics)

    logs_file.write(f" ---------- method = {method} ---------- ")
    logs_file.write(str(metrics))

    print(
        f"% Times that the context challenge was from the same source as the queried challenge: {same_category / total * 100}",
        file=logs_file)

    logs_file.flush()


if __name__ == "__main__":
    device = torch.device('cuda')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_checkpoint = BertModel.from_pretrained('bert-base-uncased').to(device)

    for k in cfg.questions_in_embedding:
        create_train_index(bert_checkpoint=bert_checkpoint,
                           bert_tokenizer=tokenizer,
                           train_data_filepath=cfg.train_data_filepath,
                           index_filepath=f'indexes/index_for_gpt_not_finetuned_bert_k={k}.pth',
                           num_questions_in_embedding=k,
                           device=device)

        index = torch.load(f'indexes/index_for_gpt_not_finetuned_bert_k={k}.pth')
        index['challenges_embeddings'] = index['challenges_embeddings'].to(device)

        train_data = json.load(open(cfg.train_data_filepath, 'r'))['data']
        dev_data = json.load(open(cfg.gpt_dev_subset_path, 'r'))['data']

        embeddings_test_bert(model=bert_checkpoint,
                             tokenizer=tokenizer,
                             index=index,
                             train_data=train_data,
                             dev_data=dev_data,
                             num_questions_in_embedding=k,
                             device=device,
                             predictions_filepath=f'../predictions/gpt_predictions_bert_not_finetuned_k={k}.json',
                             logs_file=cfg.logs_file)

    #     # TODO ---- CHANGE WHEN SWITCHING TO ANOTHER EXAMPLE CHOOSING METHODS !

    # train_data = json.load(open(cfg.train_data_filepath, 'r'))['data']
    # dev_data = json.load(open(cfg.gpt_dev_subset_path, 'r'))['data']
    # 
    # for i in range(0, 5):
    #     random_test(train_data=train_data,
    #                 dev_data=dev_data,
    #                 predictions_filepath=f'../predictions/gpt_predictions_same_source_random_t={i}',
    #                 logs_file=cfg.logs_file,
    #                 method='same_source')
