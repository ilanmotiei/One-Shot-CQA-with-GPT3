
import json


def ds_stats(ds_filepath):
    dataset = json.load(open(ds_filepath, 'r'))['data']

    challenges_lens = [len(c['questions']) for c in dataset]

    print("------------------------------")
    print(f"DS Filepath: {ds_filepath}")
    print(f"DF Length: {len(dataset)}")
    print(f"Average Challenge Length: {sum(challenges_lens) / len(challenges_lens)}")


if __name__ == "__main__":
    ds_stats('coqa-train-v1.0.json')