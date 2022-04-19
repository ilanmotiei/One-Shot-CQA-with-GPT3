import torch
from torch import nn
from transformers import BertModel, BertTokenizer
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from cqa.utils import Sample, token_ids_to_sentence, Challenge
from cqa.dataset import CQADataset


class CQAer(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.bert = BertModel.from_pretrained(args.bert_model)

        if not args.fine_tune_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        self.bert_encoding_dim = args.bert_encoding_dim
        self.bert_max_input_len = args.bert_max_input_len
        self.k = args.history_window_size
        self.device = args.device
        self.include_in_span_layer = args.include_in_span_layer

        self.bi_gru_1 = nn.GRU(input_size=(2 * self.k + 1) * self.bert_encoding_dim,
                               hidden_size=self.bert_encoding_dim,
                               num_layers=1,
                               bidirectional=True,
                               batch_first=True).to(self.device)

        self.linear_layer_1 = nn.Linear(in_features=(2 * self.k + 3) * self.bert_encoding_dim,
                                        out_features=1).to(self.device)
        self.linear_layer_2 = nn.Linear(in_features=(2 * self.k + 3) * self.bert_encoding_dim,
                                        out_features=1).to(self.device)
        self.linear_layer_3 = nn.Linear(in_features=(2 * self.k + 3) * self.bert_encoding_dim,
                                        out_features=4).to(self.device)

        self.bi_gru_2 = nn.GRU(input_size=2 * self.bert_encoding_dim,
                               hidden_size=self.bert_encoding_dim,
                               num_layers=1,
                               bidirectional=True,
                               batch_first=True).to(self.device)

        if self.include_in_span_layer:
            self.in_span_layer = nn.Linear(in_features=self.bert_encoding_dim, out_features=1)
            # ^ : for predicting which words at the context are in the answer span and which doesn't.
            # we only use this for back-propagating gradients and not for prediction.
        else:
            self.in_span_layer = None

    def encode(self, ids, masks, token_types, context_start_idx, context_end_idx, max_passage_len):
        # ids.shape == masks.shape = token_types.shape == (N, self.bert_max_input_len)

        encodings, _ = self.bert(ids,
                                 masks,
                                 token_types,
                                 return_dict=False)

        # encodings.shape = (N, self.bert_max_input_len, self.bert_encoding_dim)

        passages_encodings = []

        for i in range(ids.size(0)):
            passage_encoding = encodings[i, context_start_idx[i].item(): context_end_idx[i].item(), :]
            # passage_encoding.shape = (T_i, self.bert_encoding_dim)
            padded_passage_encoding = torch.cat([passage_encoding, torch.zeros(size=(max_passage_len-passage_encoding.size(0), self.bert_encoding_dim),
                                                                               device=self.device)],
                                                dim=0)
            passages_encodings.append(padded_passage_encoding)

        # padded = pad_sequence(passages_encodings, batch_first=True)  # shape = (N, max_i(T_i), self.bert_encoding_dim)
        padded = torch.stack(passages_encodings)

        return padded  # shape = (N, max_i(T_i), self.bert_encoding_dim)

    def forward(self, batch, use_gold_answers, return_predicted_answers=False, return_confidence_scores=False):
        answer_span_start, answer_span_end, \
            question_n_context_ids, question_n_context_mask, question_n_context_token_types, \
            question_n_context_context_start_idx, question_n_context_context_end_idx, \
            answer_n_context_ids, answer_n_context_mask, answer_n_context_token_types, \
            answer_n_context_context_start_idx, answer_n_context_context_end_idx, \
            answer_type \
            = batch

        N = question_n_context_ids.size(0)
        L = question_n_context_ids.size(1)
        max_context_length = torch.max(question_n_context_context_end_idx - question_n_context_context_start_idx).item()
        # ^ : denoted also as: max_i(len(P_i))

        # REPLACED: question_history = torch.zeros(size=(N, 0, max_context_length)).to(self.device)
        question_history = torch.zeros(size=(N, (self.k + 1) * self.bert_encoding_dim, max_context_length),
                                       device=self.device)
        # REPLACED: answer_history = torch.empty(size=(N, 0, max_context_length)).to(self.device)

        answer_history = torch.zeros(size=(N, self.k * self.bert_encoding_dim, max_context_length),
                                     device=self.device)

        total_p_s_logits = []
        total_p_e_logits = []
        total_p_answer_logits = []
        predicted_answers_ids = {}  # returning this only when 'use_gold_answer' is off
        # ^ : key is a tuple of (challenge_idx, turn_idx)
        
        total_confidence_scores = []

        if self.include_in_span_layer:
            total_in_span_logits = []
        else:
            total_in_span_logits = None

        for i in range(L):
            curr_question_encoding = self.encode(ids=question_n_context_ids[:, i, :],
                                                 masks=question_n_context_mask[:, i, :],
                                                 token_types=question_n_context_token_types[:, i, :],
                                                 context_start_idx=question_n_context_context_start_idx[:, i],
                                                 context_end_idx=question_n_context_context_end_idx[:, i],
                                                 max_passage_len=max_context_length)
            # curr_question_encoding.shape = (N, max_i(len(P_i)), self.bert_encoding_dim)

            if self.include_in_span_layer:
                curr_in_span_logits = self.in_span_layer(curr_question_encoding).squeeze(dim=2)  # (N, max_i(len_P_i)))
                total_in_span_logits.append(curr_in_span_logits)

            curr_question_encoding = curr_question_encoding.transpose(dim0=1, dim1=2)
            # curr_question_encoding.shape = (N, self.bert_encoding_dim, max_i(len(P_i)))

            # question_history.shape = (N, min(i, self.k) * self.bert_encoding_dim, max_i(len(P_i)))
            # REPLACED: question_history = torch.cat([question_history[:, max(0, i > self.k) * self.bert_encoding_dim:, :], curr_question_encoding], dim=1)
            question_history = torch.cat([question_history[:, 1 * self.bert_encoding_dim:, :], curr_question_encoding], dim=1)
            # question_history.shape = (N, (min(i, self.k) + 1) * self.bert_encoding_dim, max_i(len(P_i)))

            G = torch.cat([answer_history, question_history], dim=1)
            # ^ : shape = (N, (2 * min(i,self.k) + 1) * self.bert_encoding_dim, max_i(len(P_i))

            G = G.transpose(dim0=1, dim1=2)
            # ^ : shape = (N, max_i(len(P_i)), (2 * min(i,self.k) + 1)) * self.bert_encoding_dim)

            G_packed = pack_padded_sequence(G,
                                            lengths=torch.max(question_n_context_context_end_idx-question_n_context_context_start_idx, dim=1).values.to('cpu'),
                                            batch_first=True,
                                            enforce_sorted=False)

            # bi_gru_1 = self.bi_grus_1[self.k]
            # REPLACED: bi_gru_1 = self.bi_grus_1[min(i, self.k)]
            M_1_packed, _ = self.bi_gru_1(G_packed)  # packed sequence
            M_1, _ = pad_packed_sequence(M_1_packed, batch_first=True)
            # M_1.shape = (N, max_i(len(P_i)), 2 * self.bert_encoding_dim)
            # REPLACED: linear_layer_1 = self.linear_layers_1[min(i, self.k)]
            # linear_layer_1 = self.linear_layers_1[self.k]
            p_s_logits = self.linear_layer_1(torch.cat([G, M_1], dim=2))  # shape = (N, self.bert_max_input_size, 1)
            p_s_logits = p_s_logits.squeeze(dim=2)

            M_2_packed, _ = self.bi_gru_2(M_1_packed)  # packed sequence
            M_2, passages_lengths = pad_packed_sequence(M_2_packed, batch_first=True)
            # M_2.shape = (N, max_i(len(P_i)), 2 * self.bert_encoding_dim)
            # REPLACED: linear_layer_2 = self.linear_layers_2[min(i, self.k)]
            # linear_layer_2 = self.linear_layers_2[self.k]
            p_e_logits = self.linear_layer_2(torch.cat([G, M_2], dim=2))  # shape = (N, max_i(len(P_i)), 1)
            p_e_logits = p_e_logits.squeeze(dim=2)  # shape = (N, max_i(len(P_i)))

            # REPLACED: linear_layer_3 = self.linear_layers_3[min(i, self.k)]
            # linear_layer_3 = self.linear_layers_3[self.k]
            p_answer_logits = self.linear_layer_3(torch.cat([G, M_2], dim=2))  # shape = (N, max_i(len(P_i)), 4)

            if use_gold_answers:
                # ------- use teacher forcing:
                curr_answer_n_context_ids = answer_n_context_ids[:, i, :]
                curr_answer_n_context_masks = answer_n_context_mask[:, i, :]
                curr_answer_n_context_token_types = answer_n_context_token_types[:, i, :]
                curr_answer_n_context_ids_context_start = answer_n_context_context_start_idx[:, i]
                curr_answer_n_context_ids_context_end = answer_n_context_context_end_idx[:, i]

                # ------- calculate the answer predicted by the model:
                if return_predicted_answers:
                    answer_start_idx_pred, answer_end_idx_pred, answer_type_pred = \
                        self.predict(p_s_logits, p_e_logits, p_answer_logits, list(passages_lengths))

                    answer_start_idx_pred = question_n_context_context_start_idx[:, i] + answer_start_idx_pred
                    answer_end_idx_pred = question_n_context_context_start_idx[:, i] + answer_end_idx_pred

                    for challenge_idx in range(N):
                        if answer_type_pred[challenge_idx] == 3:
                            # Answer is a Span
                            curr_answer_ids = list(question_n_context_ids[challenge_idx, i, answer_start_idx_pred[challenge_idx]: answer_end_idx_pred[challenge_idx]+1])
                        elif answer_type_pred[challenge_idx] == 2:
                            # Answer is "Unknown"
                            curr_answer_ids = self.bert_tokenizer.encode("unknown")[1:-1]
                        elif answer_type_pred[challenge_idx] == 1:
                            # Answer is "No"
                            curr_answer_ids = self.bert_tokenizer.encode("no")[1:-1]
                        elif answer_type_pred[challenge_idx] == 0:
                            # Answer is "Yes"
                            curr_answer_ids = self.bert_tokenizer.encode("yes")[1:-1]

                        predicted_answers_ids[(challenge_idx, i)] = curr_answer_ids
            else:
                # ------- predict the answer for using it at the history of the next question:
                if return_confidence_scores:
                    answer_start_idx_pred, answer_end_idx_pred, answer_type_pred, confidence_scores = \
                        self.predict(p_s_logits, p_e_logits, p_answer_logits, list(passages_lengths), return_confidence_scores)
                    total_confidence_scores += [confidence_scores]
                else:
                    answer_start_idx_pred, answer_end_idx_pred, answer_type_pred = \
                        self.predict(p_s_logits, p_e_logits, p_answer_logits, list(passages_lengths))

                answer_start_idx_pred = question_n_context_context_start_idx[:, i] + answer_start_idx_pred
                answer_end_idx_pred = question_n_context_context_start_idx[:, i] + answer_end_idx_pred

                curr_answer_n_context_ids = []
                curr_answer_n_context_masks = []
                curr_answer_n_context_token_types = []
                curr_answer_n_context_ids_context_start = []
                curr_answer_n_context_ids_context_end = []

                for challenge_idx in range(N):
                    if answer_type_pred[challenge_idx] == 3:
                        # Answer is a Span
                        curr_answer_ids = list(question_n_context_ids[challenge_idx, i, answer_start_idx_pred[challenge_idx]: answer_end_idx_pred[challenge_idx]+1])
                    elif answer_type_pred[challenge_idx] == 2:
                        # Answer is "Unknown"
                        curr_answer_ids = self.bert_tokenizer.encode("unknown")[1:-1]
                    elif answer_type_pred[challenge_idx] == 1:
                        # Answer is "No"
                        curr_answer_ids = self.bert_tokenizer.encode("no")[1:-1]
                    elif answer_type_pred[challenge_idx] == 0:
                        # Answer is "Yes"
                        curr_answer_ids = self.bert_tokenizer.encode("yes")[1:-1]

                    predicted_answers_ids[(challenge_idx, i)] = curr_answer_ids

                    context_ids = question_n_context_ids[challenge_idx, i,
                                  question_n_context_context_start_idx[challenge_idx, i]: question_n_context_context_end_idx[challenge_idx, i]]
                    context_ids = list(context_ids)

                    curr_answer_n_context_total_len = (len(curr_answer_ids) + 2) + (len(context_ids) + 1)

                    if curr_answer_n_context_total_len > self.bert_max_input_len:
                        # cut a part from the answer ids
                        curr_answer_ids = curr_answer_ids[:-(curr_answer_n_context_total_len - self.bert_max_input_len)]

                    curr_challenge_answer_n_context_ids = [self.bert_tokenizer.cls_token_id] + curr_answer_ids + \
                                                          [self.bert_tokenizer.sep_token_id] + \
                                                          context_ids + [self.bert_tokenizer.sep_token_id]

                    curr_challenge_answer_n_context_token_types = (2 + len(curr_answer_ids)) * [0] + \
                                                                  (len(context_ids) + 1) * [1]
                    curr_challenge_answer_n_context_token_types += [0] * (self.bert_max_input_len - len(curr_challenge_answer_n_context_token_types))

                    curr_challenge_answer_n_context_masks = len(curr_challenge_answer_n_context_ids) * [1] + [0] * (
                                self.bert_max_input_len - len(curr_challenge_answer_n_context_ids))

                    curr_challenge_answer_n_context_ids = curr_challenge_answer_n_context_ids + \
                                                          [0] * (self.bert_max_input_len - len(curr_challenge_answer_n_context_ids))

                    curr_answer_n_context_ids.append(torch.Tensor(curr_challenge_answer_n_context_ids))
                    curr_answer_n_context_masks.append(torch.Tensor(curr_challenge_answer_n_context_masks))
                    curr_answer_n_context_token_types.append(torch.Tensor(curr_challenge_answer_n_context_token_types))
                    curr_answer_n_context_ids_context_start.append(len(curr_answer_ids) + 2)
                    curr_answer_n_context_ids_context_end.append(len(curr_answer_ids) + 2 + len(context_ids))

                curr_answer_n_context_ids = torch.stack(curr_answer_n_context_ids).long().to(self.device)
                curr_answer_n_context_masks = torch.stack(curr_answer_n_context_masks).int().to(self.device)
                curr_answer_n_context_token_types = torch.stack(curr_answer_n_context_token_types).int().to(self.device)
                curr_answer_n_context_ids_context_start = torch.Tensor(curr_answer_n_context_ids_context_start).long().to(self.device)
                curr_answer_n_context_ids_context_end = torch.Tensor(curr_answer_n_context_ids_context_end).long().to(self.device)

            # curr_answer_n_context_ids.shape == curr_answer_n_context_masks.shape == \
            # curr_answer_n_context_token_types.shape == (N, self.bert_max_input_len)

            curr_answer_encoding = self.encode(curr_answer_n_context_ids,
                                               curr_answer_n_context_masks,
                                               curr_answer_n_context_token_types,
                                               curr_answer_n_context_ids_context_start,
                                               curr_answer_n_context_ids_context_end,
                                               max_passage_len=max_context_length)

            # curr_answer_encoding.shape = (N, max_i(len(P_i)), self.bert_encoding_dim)
            curr_answer_encoding = curr_answer_encoding.transpose(dim0=1, dim1=2)
            # curr_answer_encoding.shape = (N, self.bert_encoding_dim, max_i(len(P_i)))

            # answer_history.shape = (N, min(i, self.k) * self.bert_encoding_dim, max_i(len(P_i)))
            # REPLACED: answer_history = torch.cat([answer_history[:, max(0, i >= self.k) * self.bert_encoding_dim:, :], curr_answer_encoding], dim=1)
            answer_history = torch.cat([answer_history[:, 1 * self.bert_encoding_dim:, :], curr_answer_encoding], dim=1)
            # answer_history.shape = (N, (min(i, self.k) + 1) * self.bert_encoding_dim, max_i(len(P_i)))

            total_p_s_logits.append(p_s_logits)
            total_p_e_logits.append(p_e_logits)
            total_p_answer_logits.append(p_answer_logits)

        total_p_s_logits = torch.stack(total_p_s_logits, dim=1)  # shape = (N, L, self.bert_max_input_len)
        total_p_e_logits = torch.stack(total_p_e_logits, dim=1)  # shape = (N, L, self.bert_max_input_len)
        total_p_answer_logits = torch.stack(total_p_answer_logits, dim=1)  # shape = (N, L, self.bert_max_input_len, 4)

        if self.include_in_span_layer:
            total_in_span_logits = torch.stack(total_in_span_logits, dim=1)  # shape = (N, L, max_passage_len)
        
        if return_confidence_scores:
            total_confidence_scores = torch.stack(total_confidence_scores, dim=1)
    
        if return_confidence_scores:
            return total_p_s_logits, total_p_e_logits, total_p_answer_logits, total_in_span_logits, predicted_answers_ids, total_confidence_scores
        else:
            return total_p_s_logits, total_p_e_logits, total_p_answer_logits, total_in_span_logits, predicted_answers_ids

    def predict(self, p_s_logits, p_e_logits, p_answer_logits, challenges_passages_lengths, return_confidence_scores=False):
        # p_s_logits.shape == p_e_logits.shape == (N, max_passage_length)
        # p_answer_logits.shape == (N, max_passsage_length, 4)
        # challenges_passages_lengths.shape == (N) == The length of each passage

        N = p_s_logits.size(0)
        max_passage_length = p_s_logits.size(1)

        answer_start_idx = torch.empty(size=(N,), dtype=torch.int).to(self.device)
        answer_end_idx = torch.empty(size=(N,), dtype=torch.int).to(self.device)

        answer_type_per_token = torch.argmax(p_answer_logits, dim=2)  # shape = (N, max_passage_length)
        final_answer_type = torch.empty(size=(N,)).to(self.device)
        
        p_answer_probs = torch.nn.functional.softmax(p_answer_logits, dim=2)  # shape = (N, max_passage_length, 4)
        p_s_probs = torch.nn.functional.softmax(p_s_logits, dim=1)  # shape = (N, max_passage_length)
        p_e_probs = torch.nn.functional.softmax(p_e_logits, dim=1)  # shape = (N, max_passage_length)
        confidence_scores = torch.empty(size=(N, )).to(self.device)

        for challenge_idx in range(N):
            argmax_i = torch.empty(size=(challenges_passages_lengths[challenge_idx], ), dtype=torch.long)
            # argmax_i[j] contains the : argmax_i<=j(p_s_logits[challenge_idx, i])
            argmax_i[0] = 0

            j_scores = torch.empty(size=(max_passage_length,))
            j_scores[0] = p_s_logits[challenge_idx, 0] + p_e_logits[challenge_idx, 0]
            argmax_j = 0

            for j in range(1, challenges_passages_lengths[challenge_idx]):
                if p_s_logits[challenge_idx, j] > p_s_logits[challenge_idx, argmax_i[j - 1]]:
                    argmax_i[j] = j
                else:
                    argmax_i[j] = argmax_i[j - 1]

                j_scores[j] = p_s_logits[challenge_idx, argmax_i[j]] + p_e_logits[challenge_idx, j]

                if j_scores[j] > j_scores[argmax_j]:
                    argmax_j = j

            answer_end_idx[challenge_idx] = argmax_j
            answer_start_idx[challenge_idx] = argmax_i[argmax_j]

            final_answer_type[challenge_idx] = answer_type_per_token[challenge_idx, argmax_j]
            
            if final_answer_type[challenge_idx] != 3:
                confidence_scores[challenge_idx] = p_answer_probs[0, challenge_idx, int(final_answer_type[challenge_idx])]
            else:
                confidence_scores[challenge_idx] = p_s_probs[0, answer_start_idx[challenge_idx]] * p_e_probs[0, answer_end_idx[challenge_idx]]
        
        if return_confidence_scores:
            return answer_start_idx, answer_end_idx, final_answer_type, confidence_scores
        else:
            return answer_start_idx, answer_end_idx, final_answer_type

    def answer(self,
               context: str,
               question: str,
               state: dict = None):
        """
        :param context: The context of the story.
        :param question: The current asked question.
        :param state: contains the current answer history and the current questions history.
        :return:
        """
        
        # TODO: Add 'with torch.no_grad()'.

        N = 1

        # preprocess

        sample = Sample(tokenizer=self.bert_tokenizer,
                        challenge_id="doesn't matter",
                        context=context,
                        question={'input_text': question, 'turn_id': 0},
                        answer={'input_text': context.split()[0], 'turn_id': 0},
                        bert_max_input_len=self.bert_max_input_len,
                        preprocessed=False)
        
        sample.question_n_context_ids = sample.question_n_context_ids.to(self.device)
        sample.question_n_context_context_start_idx = sample.question_n_context_context_start_idx.to(self.device)
        sample.question_n_context_context_end_idx = sample.question_n_context_context_end_idx.to(self.device)
        sample.question_n_context_token_types = sample.question_n_context_token_types.to(self.device)
        sample.question_n_context_mask = sample.question_n_context_mask.to(self.device)

        max_context_length = int(sample.question_n_context_context_end_idx - sample.question_n_context_context_start_idx)

        # initialize state if wasn't initialized

        if state is None:
            state = {
                'question_history': torch.zeros(size=(N, (self.k + 1) * self.bert_encoding_dim, max_context_length),
                                                 device=self.device),
                'answer_history': torch.zeros(size=(N, self.k * self.bert_encoding_dim, max_context_length),
                                               device=self.device)
            }

        # compute model's logits

        curr_question_encoding = self.encode(ids=sample.question_n_context_ids.unsqueeze(0).long(),
                                             masks=sample.question_n_context_mask.unsqueeze(0).long(),
                                             token_types=sample.question_n_context_token_types.unsqueeze(0).long(),
                                             context_start_idx=sample.question_n_context_context_start_idx.unsqueeze(0).int(),
                                             context_end_idx=sample.question_n_context_context_end_idx.unsqueeze(0).int(),
                                             max_passage_len=max_context_length)
        # curr_question_encoding.shape = (N, max_i(len(P_i)), self.bert_encoding_dim)

        curr_question_encoding = curr_question_encoding.transpose(dim0=1, dim1=2)
        # curr_question_encoding.shape = (N, self.bert_encoding_dim, max_i(len(P_i)))

        # question_history.shape = (N, min(i, self.k) * self.bert_encoding_dim, max_i(len(P_i)))
        state['question_history'] = torch.cat([state['question_history'][:, 1 * self.bert_encoding_dim:, :],
                                               curr_question_encoding], dim=1)
        # question_history.shape = (N, (min(i, self.k) + 1) * self.bert_encoding_dim, max_i(len(P_i)))

        G = torch.cat([state['answer_history'], state['question_history']], dim=1)
        # ^ : shape = (N, (2 * min(i,self.k) + 1) * self.bert_encoding_dim, max_i(len(P_i))

        G = G.transpose(dim0=1, dim1=2)
        # ^ : shape = (N, max_i(len(P_i)), (2 * min(i,self.k) + 1)) * self.bert_encoding_dim)

        G_packed = pack_padded_sequence(G,
                                        lengths=(sample.question_n_context_context_end_idx -
                                                 sample.question_n_context_context_start_idx).int().to('cpu'),
                                        batch_first=True,
                                        enforce_sorted=False)

        M_1_packed, _ = self.bi_gru_1(G_packed)
        M_1, _ = pad_packed_sequence(M_1_packed, batch_first=True)
        # M_1.shape = (N, max_i(len(P_i)), 2 * self.bert_encoding_dim)
        p_s_logits = self.linear_layer_1(torch.cat([G, M_1], dim=2))  # shape = (N, self.bert_max_input_size, 1)
        p_s_logits = p_s_logits.squeeze(dim=2)

        M_2_packed, _ = self.bi_gru_2(M_1_packed)
        M_2, passages_lengths = pad_packed_sequence(M_2_packed, batch_first=True)
        # M_2.shape = (N, max_i(len(P_i)), 2 * self.bert_encoding_dim)
        p_e_logits = self.linear_layer_2(torch.cat([G, M_2], dim=2))  # shape = (N, max_i(len(P_i)), 1)
        p_e_logits = p_e_logits.squeeze(dim=2)  # shape = (N, max_i(len(P_i)))

        p_answer_logits = self.linear_layer_3(torch.cat([G, M_2], dim=2))  # shape = (N, max_i(len(P_i)), 4)

        # predict:

        answer_start_idx_pred_a, answer_end_idx_pred_a, answer_type_pred = \
            self.predict(p_s_logits, p_e_logits, p_answer_logits, list(passages_lengths))

        answer_start_idx_pred_b = sample.question_n_context_context_start_idx.unsqueeze(0) + answer_start_idx_pred_a
        answer_end_idx_pred_b = sample.question_n_context_context_start_idx.unsqueeze(0) + answer_end_idx_pred_a

        # update the state:

        p_s_probs = torch.nn.functional.softmax(p_s_logits, dim=1)
        p_e_probs = torch.nn.functional.softmax(p_e_logits, dim=1)
        p_answer_probs = torch.nn.functional.softmax(p_answer_logits, dim=2)
        
        if answer_type_pred[0] == 3:
            # Answer is a Span
            curr_answer_ids = list(sample.question_n_context_ids[answer_start_idx_pred_b[0]:
                                                                     answer_end_idx_pred_b[0]+1])
            confidence_score = p_s_probs[0][answer_start_idx_pred_a[0]] * \
                               p_e_probs[0][answer_end_idx_pred_a[0]]
        elif answer_type_pred[0] == 2:
            # Answer is "Unknown"
            curr_answer_ids = self.bert_tokenizer.encode("unknown")[1:-1]
            confidence_score = p_answer_probs[0][answer_end_idx_pred_a[0]][2]
        elif answer_type_pred[0] == 1:
            # Answer is "No"
            curr_answer_ids = self.bert_tokenizer.encode("no")[1:-1]
            confidence_score = p_answer_probs[0][answer_end_idx_pred_a[0]][1]
        elif answer_type_pred[0] == 0:
            # Answer is "Yes"
            curr_answer_ids = self.bert_tokenizer.encode("yes")[1:-1]
            confidence_score = p_answer_probs[0][answer_end_idx_pred_a[0]][0]

        context_ids = sample.question_n_context_ids[int(sample.question_n_context_context_start_idx):
                                                    int(sample.question_n_context_context_end_idx)]
        context_ids = list(context_ids)

        curr_answer_n_context_total_len = (len(curr_answer_ids) + 2) + (len(context_ids) + 1)

        if curr_answer_n_context_total_len > self.bert_max_input_len:
            # cut a part from the answer ids
            curr_answer_ids = curr_answer_ids[:-(curr_answer_n_context_total_len - self.bert_max_input_len)]

        curr_challenge_answer_n_context_ids = [self.bert_tokenizer.cls_token_id] + curr_answer_ids + \
                                              [self.bert_tokenizer.sep_token_id] + \
                                              context_ids + [self.bert_tokenizer.sep_token_id]

        curr_challenge_answer_n_context_token_types = (2 + len(curr_answer_ids)) * [0] + \
                                                      (len(context_ids) + 1) * [1]
        curr_challenge_answer_n_context_token_types += [0] * (self.bert_max_input_len - len(curr_challenge_answer_n_context_token_types))

        curr_challenge_answer_n_context_masks = len(curr_challenge_answer_n_context_ids) * [1] + [0] * (
                self.bert_max_input_len - len(curr_challenge_answer_n_context_ids))

        curr_challenge_answer_n_context_ids = curr_challenge_answer_n_context_ids + \
                                              [0] * (self.bert_max_input_len - len(curr_challenge_answer_n_context_ids))

        curr_answer_n_context_ids = torch.Tensor(curr_challenge_answer_n_context_ids).unsqueeze(0).long().to(self.device)
        curr_answer_n_context_masks = torch.Tensor(curr_challenge_answer_n_context_masks).unsqueeze(0).long().to(self.device)
        curr_answer_n_context_token_types = torch.Tensor(curr_challenge_answer_n_context_token_types).unsqueeze(0).long().to(self.device)
        curr_answer_n_context_ids_context_start = torch.Tensor([len(curr_answer_ids) + 2]).long().to(self.device)
        curr_answer_n_context_ids_context_end = torch.Tensor([len(curr_answer_ids) + 2 + len(context_ids)]).long().to(self.device)

        curr_answer_encoding = self.encode(curr_answer_n_context_ids,
                                           curr_answer_n_context_masks,
                                           curr_answer_n_context_token_types,
                                           curr_answer_n_context_ids_context_start,
                                           curr_answer_n_context_ids_context_end,
                                           max_passage_len=max_context_length)

        # curr_answer_encoding.shape = (N, max_i(len(P_i)), self.bert_encoding_dim)
        curr_answer_encoding = curr_answer_encoding.transpose(dim0=1, dim1=2)
        # curr_answer_encoding.shape = (N, self.bert_encoding_dim, max_i(len(P_i)))

        # answer_history.shape = (N, min(i, self.k) * self.bert_encoding_dim, max_i(len(P_i)))
        state['answer_history'] = torch.cat([state['answer_history'][:, 1 * self.bert_encoding_dim:, :],
                                             curr_answer_encoding], dim=1)
        # answer_history.shape = (N, (min(i, self.k) + 1) * self.bert_encoding_dim, max_i(len(P_i)))

        return token_ids_to_sentence(tokenizer=self.bert_tokenizer, token_ids=curr_answer_ids), confidence_score
    
    
    def answer1(self,
               context: str,
               question: str,
               state: dict = None):
        """
        :param context: The context of the story.
        :param question: The current asked question.
        :param state: contains the current answer history and the current questions history.
        :return:
        """
        
        def initial_state(context, question):
            state = {
                'id': "doesn't matter",
                'story': context,
                'questions': [
                    {
                        'input_text': question,
                        'turn_id': "doesn't matter"
                    }
                ],
                'answers': [
                    {
                        'input_text': context.split()[0],
                        'span_text': context.split()[0],
                        'turn_id': "doesn't matter"
                    }
                ]
            }
            
            return state
        
        if state is None:
            state = initial_state(context, question)
        else:
            if state['story'] != context:
                state = initial_state(context, question)
            else:
                if len(state['questions']) >= self.k:
                    state['questions'] = state['questions'][1:] + [{'input_text': question, 'turn_id': 'dont matter'}]
                    state['answers'] = state['answers'][1:] + [{'input_text': context.split()[0], 'span_text': context.split()[0], 'turn_id': "Doesn't matter"}]
                else:
                    state['questions'] += [{'input_text': question, 'turn_id': "Doesn't matter"}]
                    state['answers'] += [{'input_text': context.split()[0], 'span_text': context.split()[0], 'turn_id': "Doesn't matter"}]

        challenge = Challenge(challenge=state,
                              tokenizer=self.bert_tokenizer,
                              bert_max_input_len=self.bert_max_input_len,
                              preprocessed=False)

        batch_data =  ((challenge.answer_span_start.long(), challenge.answer_span_end.long(),
                        challenge.question_n_context_ids.long(), challenge.question_n_context_mask.long(),
                        challenge.question_n_context_token_types.long(),
                        challenge.question_n_context_context_start_idx.int(),
                        challenge.question_n_context_context_end_idx.int(),
                        challenge.answer_n_context_ids.long(), challenge.answer_n_context_mask.long(),
                        challenge.answer_n_context_token_types.long(),
                        challenge.answer_n_context_context_start_idx.long(), challenge.answer_n_context_context_end_idx.long(),
                        challenge.category.long()), challenge.id, challenge.turn_ids)
        
        batch_data, lengths, ids, turn_ids = CQADataset.collate_fn([batch_data])

        with torch.set_grad_enabled(False):
            batch_data = [t.to(self.device) for t in batch_data]
            p_s_logits, p_e_logits, p_answer_logits, _, predicted_answers_ids, confidence_scores = self(batch_data,
                                                                                                        use_gold_answers=False,
                                                                                                        return_predicted_answers=True,
                                                                                                        return_confidence_scores=True)
            
            answer = token_ids_to_sentence(self.bert_tokenizer, predicted_answers_ids[(0, int(lengths[0] - 1))])
            confidence_score = confidence_scores[0, -1]
        
            state['answers'][-1]['input_text'] = answer
            state['answers'][-1]['span_text'] = answer
            
        return state, answer, confidence_score.item()
