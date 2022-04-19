
import sys
sys.path.append('..')
from cqa.model import CQAer
import json
from cqa.official_evaluation_script import calculate_metrics

predictions_file = '../predictions/gpt_predictions_bert_finetuned_k=17.json'

print(calculate_metrics(data_filepath='../data/coqa_dev_subset_for_gpt.json', predictions_filepath=predictions_file))