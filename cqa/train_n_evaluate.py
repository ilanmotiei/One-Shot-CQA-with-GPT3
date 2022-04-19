
import torch
from dataset import CQADataset
from torch.utils.data import DataLoader
from torch import nn
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import StepLR
from model import CQAer
import json
from utils import token_ids_to_sentence
import os
import argparse
from losses import padded_passages_logits_loss, in_span_loss_with_logits
from official_evaluation_script import calculate_metrics
from os import path
from scheduler import CostumeScheduler
import tqdm
import pickle


def train_epoch(model, epoch, optimizer, tokenizer, train_dataloader, validation_dataloader, args):
    print(f" --------------- Epoch {epoch + 1}/{args.epochs} Training Start --------------- ", file=args.logs_file)

    span_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    type_criterion = nn.CrossEntropyLoss(ignore_index=-1)
    in_span_criterion = nn.BCEWithLogitsLoss(reduction='none')

    epoch_loss = 0
    epoch_span_start_loss = 0
    epoch_span_end_loss = 0
    epoch_type_loss = 0
    if args.include_in_span_layer:
        epoch_in_span_loss = 0

    model.train()

    with torch.set_grad_enabled(True):
        for batch_idx, (batch_data, lengths, ids, turn_ids) in tqdm.tqdm(enumerate(train_dataloader)):
            batch_data = [t.to(args.device) for t in batch_data]

            p_s_logits, p_e_logits, p_answer_logits, in_span_logits, _ = model(batch_data, use_gold_answers=True)

            answer_span_start, answer_span_end, \
                question_n_context_ids, question_n_context_mask, question_n_context_token_types, \
                question_n_context_context_start_idx, question_n_context_context_end_idx, \
                answer_n_context_ids, answer_n_context_mask, answer_n_context_token_types, \
                answer_n_context_context_start_idx, answer_n_context_context_end_idx, \
                answer_type \
                = batch_data\

            max_passage_len = p_e_logits.size(2)

            span_start_loss = padded_passages_logits_loss(p_s_logits, answer_span_start, span_criterion)
            span_end_loss = padded_passages_logits_loss(p_e_logits, answer_span_end, span_criterion)

            type_loss = type_criterion(p_answer_logits.transpose(dim0=1, dim1=3).transpose(dim0=2, dim1=3),
                                       torch.stack([answer_type] * max_passage_len, dim=2))

            if args.include_in_span_layer:
                in_span_loss = in_span_loss_with_logits(in_span_logits=in_span_logits,
                                                        answer_span_start=answer_span_start,
                                                        answer_span_end=answer_span_end,
                                                        lengths=lengths,
                                                        criterion=in_span_criterion,
                                                        device=args.device)
            else:
                in_span_loss = 0

            if optimizer._step % args.in_span_loss_lambda_decay_steps == 0:
                args.in_span_loss_lambda *= args.in_span_loss_lambda_decay_gamma

            total_loss = args.span_start_lambda * span_start_loss + \
                         args.span_end_lambda * span_end_loss + \
                         args.type_loss_lambda * type_loss + \
                         args.in_span_loss_lambda * in_span_loss

            total_loss = total_loss / args.update_every_n_batches
            total_loss.backward()

            if ((batch_idx + 1) % args.update_every_n_batches == 0) or (batch_idx == len(train_dataloader) - 1):
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

            epoch_loss += total_loss.item() * args.update_every_n_batches
            epoch_span_start_loss += span_start_loss.item()
            epoch_span_end_loss += span_end_loss.item()
            epoch_type_loss += type_loss.item()
            if args.include_in_span_layer:
                epoch_in_span_loss += in_span_loss.item()

            if (batch_idx + 1) % args.train_print_every == 0:
                print(f" ------------ Epoch {epoch + 1}/{args.epochs} Batch {batch_idx + 1}/{len(train_dataloader)} Training Results ------------ ", file=args.logs_file)
                print(f"Total Loss: {epoch_loss / (batch_idx + 1)}", file=args.logs_file)
                print(f"Span Start Loss: {epoch_span_start_loss / (batch_idx + 1)}", file=args.logs_file)
                print(f"Span End Loss: {epoch_span_end_loss / (batch_idx + 1)}", file=args.logs_file)
                print(f"Type Loss: {epoch_type_loss / (batch_idx + 1)}", file=args.logs_file)
                if args.include_in_span_layer:
                    print(f"In-span Loss: {epoch_in_span_loss / (batch_idx + 1)}", file=args.logs_file)

                args.logs_file.flush()

            if (batch_idx + 1) % args.checkpoint_every == 0:
                model.eval()

                validate(model, tokenizer, epoch, validation_dataloader, args)
                torch.save(model.state_dict(), f'{args.models_dir}/model_epoch={epoch + 1}.pth')

                model.train()

    epoch_loss /= len(train_dataloader)
    epoch_span_start_loss /= len(train_dataloader)
    epoch_span_end_loss /= len(train_dataloader)
    epoch_type_loss /= len(train_dataloader)
    if args.include_in_span_layer:
        epoch_in_span_loss /= len(train_dataloader)

    print(f" --------------------- Epoch {epoch + 1}/{args.epochs} Final Training Results ------------------------ ", file=args.logs_file)
    print(f"Total Loss: {epoch_loss}", file=args.logs_file)
    print(f"Span Start Loss: {epoch_span_start_loss}", file=args.logs_file)
    print(f"Span End Loss: {epoch_span_end_loss}", file=args.logs_file)
    print(f"Type Loss: {epoch_type_loss}", file=args.logs_file)
    if args.include_in_span_layer:
        print(f"In-span Loss: {epoch_in_span_loss}", file=args.logs_file)

    args.logs_file.flush()


def validate(model, tokenizer, epoch, validation_dataloader, args):
    print(f" --------------- Epoch {epoch + 1}/{args.epochs} Validation Start --------------- ", file=args.logs_file)

    predictions = []

    with torch.set_grad_enabled(False):
        for batch_idx, (batch_data, lengths, ids, turn_ids) in tqdm.tqdm(enumerate(validation_dataloader)):
            batch_data = [t.to(args.device) for t in batch_data]

            p_s_logits, p_e_logits, p_answer_logits, _, predicted_answers_ids = model(batch_data,
                                                                                      use_gold_answers=False,
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

    if not path.exists(args.predictions_dir):
        os.mkdir(args.predictions_dir)

    curr_epoch_predictions_filepath = f'{args.predictions_dir}/predictions_epoch={epoch}.json'
    json.dump(predictions, fp=open(curr_epoch_predictions_filepath, 'w'), indent=2)
    metrics = calculate_metrics(data_filepath=args.dev_data_filepath,
                                predictions_filepath=curr_epoch_predictions_filepath)

    print("Results: --------------- ", file=args.logs_file)
    print(metrics, file=args.logs_file)
    args.logs_file.flush()


def train(args):
    model = CQAer(args=args).to(args.device)

    if not (args.model_checkpoint is None):
        model.load_state_dict(torch.load(args.model_checkpoint, map_location=args.device))

    tokenizer = model.bert_tokenizer

    train_dataset = CQADataset(coqa_dataset_filepath=args.train_data_filepath,
                               tokenizer=tokenizer,
                               max_challenges=args.train_dataset_max_challenges,
                               max_challenge_len=args.max_challenge_len,
                               args=args)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  collate_fn=CQADataset.collate_fn)

    optimizer = AdamW(params=model.parameters(),
                      lr=args.lr,
                      weight_decay=args.weight_decay)

    training_steps = args.epochs * (len(train_dataloader) / args.update_every_n_batches)

    optimizer = CostumeScheduler(warmup_steps=(args.warmup_steps_pct / 100) * training_steps,
                                 lr_upper_bound=args.lr,
                                 lr_decay_steps=args.lr_decay_steps,
                                 lr_decay_gamma=args.lr_decay_gamma,
                                 optimizer=optimizer)

    if not (args.optimizer_checkpoint is None):
        optimizer.load_state_dict(torch.load(args.optimizer_checkpoint, map_location=args.device))

    validation_dataset = CQADataset(coqa_dataset_filepath=args.dev_data_filepath,
                                    tokenizer=tokenizer,
                                    max_challenges=args.dev_dataset_max_challenges,
                                    max_challenge_len=args.max_challenge_len,
                                    args=args)

    validation_dataloader = DataLoader(dataset=validation_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       num_workers=args.num_workers,
                                       collate_fn=CQADataset.collate_fn)

    for epoch_idx in range(args.current_epoch_num, args.current_epoch_num + args.epochs):
        train_epoch(model,
                    epoch_idx,
                    optimizer,
                    tokenizer=tokenizer,
                    train_dataloader=train_dataloader,
                    validation_dataloader=validation_dataloader,
                    args=args)

        if not path.exists(args.models_dir):
            os.mkdir(args.models_dir)

        torch.save(model.state_dict(), f'{args.models_dir}/model_epoch={epoch_idx + 1}.pth')
        torch.save(optimizer.state_dict(), f'{args.models_dir}/optimizer_epoch={epoch_idx + 1}.pth')

        validate(model, tokenizer, epoch_idx, validation_dataloader, args)


def parse_args():
    parser = argparse.ArgumentParser('Multi-Turn Context with BERT for Conversational QA')

    parser.add_argument('--train-data-filepath', dest='train_data_filepath', default='coqa-train-v1.0.json',
                        help='CO-QA Training JSON file')

    parser.add_argument('--train-dataset-max-challenges', dest='train_dataset_max_challenges', default=-1, type=int,
                        help='Amount of challenges to use for training from the training dataset. ' +
                             '-1 for using the full training dataset for training')

    parser.add_argument('--dev-data-filepath', dest='dev_data_filepath', default='coqa-dev-v1.0.json',
                        help='CO-QA Dev JSON file', )

    parser.add_argument('--models-dir', dest='models_dir', default='models',
                        help='The directory in which we will save all the checkpoints of the model')

    parser.add_argument('--dev-dataset-max-challenges', dest='dev_dataset_max_challenges', default=-1, type=int,
                        help='Amount of challenges to se for evaluating the model from the dev dataset. ' +
                        '-1 for using all the dev dataset for evaluation')

    parser.add_argument('--max-challenge-len', dest='max_challenge_len', default=5, type=int,
                        help='Maximum amount of turns computed at each forward pass ** ! -- DURING TRAINING -- ! **')

    parser.add_argument('--evaluation-script-filepath', dest='evaluation_script_filepath',
                        default='official_evaluation_script.py', help='Official CO-QA evaluation script')

    parser.add_argument('--predictions-dir', dest='predictions_dir', default='predictions',
                        help='The direcrtory in which we will save all the JSON files containing the '
                             'predictions of the model on every epoch')

    parser.add_argument('--logs-file', dest='logs_file', default='logs.txt',
                        help='The file to print the stats at the training to')

    parser.add_argument('--bert-model', dest='bert_model', default='bert-base-uncased',
                        help='The name of the BERT model from the transformers library to use')

    parser.add_argument('--bert-max-input-len', dest='bert_max_input_len', default=512, type=int,
                        help='The maximum input length that the BERT model can get')

    parser.add_argument('--bert-encoding-dim', dest='bert_encoding_dim', default=768, type=int,
                        help='The dimension of the encodings that BERT outputs')

    parser.add_argument('--include-in-span-layer', dest='include_in_span_layer', default=False, type=bool,
                        help='Whether to include the in-span layer and loss or not')

    parser.add_argument('--history-window-size', dest='history_window_size', default=3, type=int,
                        help='The amount of turns the model considers when predicting the answer for the current turn')

    parser.add_argument('--epochs', dest='epochs', default=100, type=int,
                        help='Amount of epochs to train the model on')

    parser.add_argument('--num-workers', dest='num_workers', default=22, type=int,
                        help='Amount of CPU threads that load the data to the GPU')

    parser.add_argument('--batch-size', dest='batch_size', default=4, type=int,
                        help='Amount of challenges processed at each forward pass')

    parser.add_argument('--lr', dest='lr', default=5e-04, type=float,
                        help='Learning Rate for the AdamW optimizer that trains the model')

    parser.add_argument('--train-print-every', dest='train_print_every', default=100, type=int,
                        help='At the training loop this is the cycle time of the printing results, in batches')

    parser.add_argument('--update-every-n-batches', dest='update_every_n_batches', default=1, type=int,
                        help='The amount of batches for which we aggregate the gradient before each optimization step')

    parser.add_argument('--device', dest='device', default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='The device on which we will train and evaluate the model')

    parser.add_argument('--span-start-lambda', dest='span_start_lambda', default=1, type=float,
                        help='The contribution of the span starting index on the total loss at training time')

    parser.add_argument('--span-end-lambda', dest='span_end_lambda', default=1, type=float,
                        help='The contribution of the span ending index on the total loss at training time')

    parser.add_argument('--type-loss-lambda', dest='type_loss_lambda', default=1, type=float,
                        help='The contribution of the type loss on the total loss at training time')

    parser.add_argument('--in-span-loss-lambda', dest='in_span_loss_lambda', default=1, type=float,
                        help='The contribution of the in-span loss on the total loss at training time')

    parser.add_argument('--fine-tune-bert', dest='fine_tune_bert', default=True, type=bool,
                        help='A boolean mentioning whether to fine-tune bert or to use its encodings as features')

    parser.add_argument('--model-checkpoint', dest='model_checkpoint', default=None, type=str,
                        help='The location of a model if you want to retrain it')

    parser.add_argument('--optimizer-checkpoint', dest='optimizer_checkpoint', default=None, type=str,
                        help='The location of the state dict of an optimizer if you want to use its state')

    parser.add_argument('--current-epoch-num', dest='current_epoch_num', type=int, default=0,
                        help='The stating epoch to consider')

    parser.add_argument('--checkpoint-every', dest='checkpoint_every', default=1000, type=int,
                        help='Amount of steps between two checkpoints')

    parser.add_argument('--weight-decay', dest='weight_decay', default=0.01, type=float,
                        help='L2 weight decay for the Adam optimizer')

    parser.add_argument('--lr-decay-gamma', dest='lr_decay_gamma', default=0.9, type=float,
                        help='The learning rate decay multiplicative factor')

    parser.add_argument('--lr-decay-steps', dest='lr_decay_steps', default=1000, type=int,
                        help='Number of steps before each multiplicative decay of the learning rate')

    parser.add_argument('--warmup-steps-pct', dest='warmup_steps_pct', default=10, type=float,
                        help='The amount of optimizing steps at the warmup')

    parser.add_argument('--preprocess-data', dest='preprocess_data', default=False, type=bool,
                        help='Mentions whether to preprocess the given data files or if they are already preprocessed')

    parser.add_argument('--in-span-loss-lambda-decay-steps', dest='in_span_loss_lambda_decay_steps', default=100, type=int,
                        help='Amount of steps between each multiplicative decay of the in-span loss')

    parser.add_argument('--in-span-loss-lambda-decay-gamma', dest='in_span_loss_lambda_decay_gamma', default=0.3, type=float,
                        help='The multiplicative factor for decaying the in-span loss')

    args = parser.parse_args()

    torch.save(args, open(f'{args.models_dir}/args.pth', 'wb'))

    args.device = torch.device(args.device)
    args.logs_file = open(args.logs_file, 'w')

    return args


if __name__ == "__main__":
    OPTS = parse_args()
    train(OPTS)
