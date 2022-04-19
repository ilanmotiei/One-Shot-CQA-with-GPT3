
import json
from collections import defaultdict
import random
import configurations as cfg
from create_embeddings import create_embedding
import torch
from cqa.model import CQAer
from transformers import BertTokenizer, BertModel
import tqdm


def is_challenge_ok(bert_checkpoint,
                    tokenizer,
                    passage,
                    questions,
                    device):

    for i in range(len(questions)):
        for j in range(len(questions)):
            if create_embedding(bert_checkpoint=bert_checkpoint,
                                tokenizer=tokenizer,
                                passage=passage,
                                questions=questions,
                                num_questions_for_embedding=i,
                                device=device,
                                from_dev=True,
                                j=j) is None:
                return False

    return True


def sample(devset_path: str,
           output_path: str,
           bert_checkpoint: BertModel,
           tokenizer: BertTokenizer) -> dict:
    """
    :param devset_path: The path to the dev-set of COQA.
    :param output_path: The path in which to save the extracted subset of the dev set.
    """
    
    devset = json.load(open(devset_path, 'r'))
    devset_challenges = devset['data']
    
    source_to_challenges = defaultdict(lambda: [])

    for challenge in tqdm.tqdm(devset_challenges):
        if is_challenge_ok(bert_checkpoint=bert_checkpoint,
                           tokenizer=tokenizer,
                           passage=challenge['story'],
                           questions=[question['input_text'] for question in challenge['questions']],
                           device=device):

            source_to_challenges[challenge['source']].append(challenge)

    subsets = dict()
    for source, challenges in source_to_challenges.items():
        num_challenges_in_subset = int(cfg.num_challenges_in_subset / len(source_to_challenges.keys()))
        print(f"source '{source}' had {len(challenges)} challenges, now has {num_challenges_in_subset}")
        subsets[source] = random.choices(challenges, k=num_challenges_in_subset)
    
    sampled_dev_subset = []
    for challenges in subsets.values():
        sampled_dev_subset += challenges
        
    devset['data'] = sampled_dev_subset
    json.dump(devset, open(output_path, 'w'), indent=2)
    
    print(f"Total length of the new sampled subset: {len(devset['data'])}")


if __name__ == "__main__":
    print("A dev subset was already created! if you want to create it again remove the exit command after this print")
    exit(1)

    device = torch.device('cuda')
    cqaer = CQAer(torch.load(open(cfg.cqaer_args_filepath, 'rb')))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_checkpoint = cqaer.bert.to(device)

    sample(devset_path=cfg.coqa_original_devset_path,
           output_path=cfg.gpt_dev_subset_path,
           bert_checkpoint=bert_checkpoint,
           tokenizer=tokenizer)
