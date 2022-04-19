
# FOR CREATING THE INDEX:
max_input_len = 512
batch_size = 20
index_filepath = 'index_for_gpt.pth'

# FOR INFERENCING WITH GPT-3:
coqa_original_devset_path = '../data/coqa-dev-v1.0.json'
gpt_dev_subset_path = '../data/coqa_dev_subset_for_gpt.json'
num_challenges_in_subset = 50
gpt_engine = 'text-ada-001'

# FOR BOTH:
questions_in_embedding = list(range(8, 20))
# ^ : == k in the paper

train_data_filepath = '../data/coqa-train-v1.0.json'
cqaer_args_filepath = '../experiment_6/models/args.pth'
logs_file = open('bert_not_finetuned_k=[8, 20]_logs.txt', 'a')
