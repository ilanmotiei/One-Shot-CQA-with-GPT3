
import json


def ds_stats(ds_filepath):
    dataset = json.load(open(ds_filepath, 'r'))['data']

    print("------------------------------")
    print(f"DS Filepath: {ds_filepath}")
    print(f"DF Length: {len(dataset)}")


if __name__ == "__main__":
    ds_stats('../data/coqa-dev-v1.0_preprocessed.json')