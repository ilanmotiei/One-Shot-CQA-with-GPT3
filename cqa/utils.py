
import torch
import sacremoses

detokenizer = sacremoses.MosesDetokenizer('en')


def token_ids_to_sentence(tokenizer, token_ids):
    answer = detokenizer.detokenize(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(token_ids)).split())

    if answer == 'yes':
        answer = 'Yes'
    elif answer == 'no':
        answer = 'No'

    formatted_answer = ''

    i = 0
    while i < len(answer):
        if answer[i: i + 2] == " \'":
            formatted_answer += "\'"
            i += 2
        else:
            formatted_answer += answer[i]
            i += 1

    return formatted_answer


def find_answer_start_idx(encoded_passage, encoded_answer):
    """
    :param encoded_passage: Encoding of the passage.
    :param encoded_answer: Encoding of the answer.
    :return: The index in which 'encoded_answer' appeared at first at the encoded passage. returns -1 if wasn't found.
    """

    for i in range(0, len(encoded_passage) - len(encoded_answer) + 1):
        if all([a == b for a, b in zip(encoded_passage[i: i + len(encoded_answer)], encoded_answer)]):
            return i

    # we didn't find the starting index
    return -1


class Challenge:
    def __init__(self, challenge, tokenizer, bert_max_input_len, preprocessed):
        self.id = challenge['id']
        self.turn_ids = []

        self.samples = []

        for question, answer in zip(challenge['questions'], challenge['answers']):
            new_sample = Sample(tokenizer=tokenizer,
                                challenge_id=challenge['id'],
                                context=challenge['story'],
                                question=question,
                                answer=answer,
                                bert_max_input_len=bert_max_input_len,
                                preprocessed=preprocessed)

            self.samples.append(new_sample)
            self.turn_ids.append(question['turn_id'])

        self.discard = any([sample.discard for sample in self.samples])

        if self.discard:
            return

        # else: batchify: denote the length of the challenge (the number of turns) by L.
        self.answer_span_start = torch.cat([s.answer_span_start for s in self.samples], dim=0)  # shape = (L)
        self.answer_span_end = torch.cat([s.answer_span_end for s in self.samples], dim=0)  # shape = (L)

        # shape of the 3 tensors bellow = (L, bert_max_input_len)
        self.question_n_context_ids = torch.stack([s.question_n_context_ids for s in self.samples])
        self.question_n_context_mask = torch.stack([s.question_n_context_mask for s in self.samples])
        self.question_n_context_token_types = torch.stack([s.question_n_context_token_types for s in self.samples])

        # shape of the 2 tensors bellow = (L)
        self.question_n_context_context_start_idx = torch.cat([s.question_n_context_context_start_idx for s in self.samples], dim=0)
        self.question_n_context_context_end_idx = torch.cat([s.question_n_context_context_end_idx for s in self.samples], dim=0)

        # shape of the 3 tensors bellow = (L, bert_max_input_len)
        self.answer_n_context_ids = torch.stack([s.answer_n_context_ids for s in self.samples])
        self.answer_n_context_mask = torch.stack([s.answer_n_context_mask for s in self.samples])
        self.answer_n_context_token_types = torch.stack([s.answer_n_context_token_types for s in self.samples])
        self.answer_n_context_context_start_idx = torch.cat([s.answer_n_context_context_start_idx for s in self.samples], dim=0)
        self.answer_n_context_context_end_idx = torch.cat([s.answer_n_context_context_end_idx for s in self.samples], dim=0)

        self.category = torch.cat([s.category for s in self.samples], dim=0)  # shape = (L)

    def __len__(self):
        return len(self.samples)


class Sample:
    def __init__(self, tokenizer, challenge_id, context, question, answer, bert_max_input_len, preprocessed):
        self.discard = False
        turn_id = question['turn_id']

        context_ids = tokenizer.encode(context)
        question_ids = tokenizer.encode(question['input_text'])
        answer_text = answer['input_text']
        answer_ids = tokenizer.encode(answer_text)

        if len(question_ids) + (len(context_ids) - 1) > bert_max_input_len:
            # print(f"id: {challenge_id}, turn_id: {turn_id} :: ",
            #       "Question + Context are too long to fit as an input to a model",
            #       f" with maximum input length of: {bert_max_input_len}", sep='')

            context_ids = context_ids[: bert_max_input_len - len(question_ids)] + [tokenizer.sep_token_id]

        answer_span_start = find_answer_start_idx(encoded_passage=context_ids[1:],
                                                  encoded_answer=answer_ids[
                                                                 1:-1])  # (we don't need the [CLS] and [SEP] tokens)

        if answer_span_start == -1:
            if answer_text == "Yes" or answer_text == "yes":
                # print(f"id: {challenge_id}, turn_id: {turn_id} :: Answer isn't a span but is Yes")
                self.category = 0  # "Yes"
            elif answer_text == "No" or answer_text == "no":
                # print(f"id: {challenge_id}, turn_id: {turn_id} :: Answer isn't a span but is No")
                self.category = 1  # "No"
            elif answer_text == "unknown":
                # print(f"id: {challenge_id}, turn_id: {turn_id} :: Answer isn't a span but is Unknown")
                self.category = 2  # "Unanswerable"
            else:
                # answer['input_text'] is none of ['Yes', 'No', 'unknown'] and it wasn't found at the context
                self.category = 3  # "Span"

                if preprocessed:
                    if answer['discard'] is True:
                        # we already processed this answer and decided to discard it. discarding it turn again...
                        self.discard = True
                    else:
                        print(f"id: {challenge_id}, turn_id: {turn_id}",
                              f":: Error: answer['discard'] is False but answer['input_text'] isn't contained at the context")
                else:
                    answer_text = answer['span_text']
                    answer_ids = tokenizer.encode(answer_text)
                    answer_span_start = find_answer_start_idx(encoded_passage=context_ids[1:],
                                                              encoded_answer=answer_ids[1:-1])

                    if answer_span_start == -1:
                        # sometimes there are answers that lack some char at the start/end to be a span of the context

                        original_answer_text = answer_text
                        upper_case_letters = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
                        lower_case_letters = [chr(c) for c in range(ord('a'), ord('z') + 1)]

                        for start_char in upper_case_letters+lower_case_letters+['']:
                            for end_char in lower_case_letters+['']:
                                answer_text = start_char + original_answer_text + end_char
                                answer_ids = tokenizer.encode(answer_text)
                                answer_span_start = find_answer_start_idx(encoded_passage=context_ids[1:],
                                                                          encoded_answer=answer_ids[1:-1])

                                if answer_span_start != -1:
                                    break

                            if answer_span_start != -1:
                                break

                        if answer_span_start == -1:
                            # we didn't succeed previously : try with 2 letters at the start:
                            for start_char1 in upper_case_letters+lower_case_letters+['']:
                                for start_char2 in upper_case_letters+lower_case_letters+['']:
                                    answer_text = start_char1 + start_char2 + original_answer_text
                                    answer_ids = tokenizer.encode(answer_text)
                                    answer_span_start = find_answer_start_idx(encoded_passage=context_ids[1:],
                                                                              encoded_answer=answer_ids[1:-1])
                                    if answer_span_start != -1:
                                        break
                                if answer_span_start != -1:
                                    break

                        if answer_span_start == -1:
                            # we didn't succeed previously : try with 2 letters at the start:
                            for start_char1 in upper_case_letters+lower_case_letters+['']:
                                for start_char2 in upper_case_letters+lower_case_letters+['']:
                                    answer_text = start_char1 + start_char2 + original_answer_text
                                    answer_ids = tokenizer.encode(answer_text)
                                    answer_span_start = find_answer_start_idx(encoded_passage=context_ids[1:],
                                                                              encoded_answer=answer_ids[1:-1])
                                    if answer_span_start != -1:
                                        break
                                if answer_span_start != -1:
                                    break

                        if answer_span_start == -1:
                            # we didn't succeed previously : try with 2 letters at the end:
                            for end_char1 in upper_case_letters+lower_case_letters+['']:
                                for end_char2 in upper_case_letters+lower_case_letters+['']:
                                    answer_text = original_answer_text + end_char1 + end_char2
                                    answer_ids = tokenizer.encode(answer_text)
                                    answer_span_start = find_answer_start_idx(encoded_passage=context_ids[1:],
                                                                              encoded_answer=answer_ids[1:-1])
                                    if answer_span_start != -1:
                                        break
                                if answer_span_start != -1:
                                    break

                        if answer_span_start == -1:
                            # we didn't succeed previously : try to remove 1-2 characters from the end:
                            answer_text = original_answer_text[:-1]
                            answer_ids = tokenizer.encode(answer_text)
                            answer_span_start = find_answer_start_idx(encoded_passage=context_ids[1:],
                                                                      encoded_answer=answer_ids[1:-1])

                            if answer_span_start == -1:
                                answer_text = original_answer_text[:-2]
                                answer_ids = tokenizer.encode(answer_text)
                                answer_span_start = find_answer_start_idx(encoded_passage=context_ids[1:],
                                                                          encoded_answer=answer_ids[1:-1])

                        if answer_span_start == -1:
                            # we didn't succeed previously : try to remove 1-2 characters from the start:
                            answer_text = original_answer_text[1:]
                            answer_ids = tokenizer.encode(answer_text)
                            answer_span_start = find_answer_start_idx(encoded_passage=context_ids[1:],
                                                                      encoded_answer=answer_ids[1:-1])

                            if answer_span_start == -1:
                                answer_text = original_answer_text[2:]
                                answer_ids = tokenizer.encode(answer_text)
                                answer_span_start = find_answer_start_idx(encoded_passage=context_ids[1:],
                                                                          encoded_answer=answer_ids[1:-1])

                        # now we'll check that again
                        if answer_span_start == -1:
                            # print(f"id: {challenge_id}, turn_id: {turn_id}",
                            #       f":: Error: answer['span_text'] is not a span of the context")
                            self.discard = True
        else:
            # answer['input_text'] was found at the context
            self.category = 3  # "Span"

        answer_span_end = (answer_span_start + (len(answer_ids) - 2) - 1) if answer_span_start != -1 else -1

        if len(answer_ids) + (len(context_ids) - 1) > bert_max_input_len:
            context_ids = context_ids[: bert_max_input_len - (len(answer_ids))] + [tokenizer.sep_token_id]

            if (answer_span_start + len(answer_ids) + 1 >= bert_max_input_len) or \
                    (answer_span_end + len(answer_ids) + 1 >= bert_max_input_len):
                self.discard = True

        self.question_n_context_ids = question_ids + context_ids[1:]
        self.question_n_context_token_types = [0] * len(question_ids) + [1] * (len(context_ids) - 1)
        self.question_n_context_context_start_idx = len(question_ids)
        self.question_n_context_context_end_idx = self.question_n_context_context_start_idx + (len(context_ids) - 2)

        self.answer_n_context_ids = answer_ids + context_ids[1:]
        self.answer_n_context_token_types = [0] * len(answer_ids) + [1] * (len(context_ids) - 1)
        self.answer_n_context_context_start_idx = len(answer_ids)
        self.answer_n_context_context_end_idx = self.answer_n_context_context_start_idx + (len(context_ids) - 2)

        self.answer_span_start = answer_span_start
        self.answer_span_end = answer_span_end

        self.question_n_context_mask = [1] * len(self.question_n_context_ids) + [0] * (bert_max_input_len - len(self.question_n_context_ids))
        self.question_n_context_ids += [0] * (bert_max_input_len - len(self.question_n_context_ids))
        self.question_n_context_token_types += [0] * (bert_max_input_len - len(self.question_n_context_token_types))

        self.answer_n_context_mask = [1] * len(self.answer_n_context_ids) + [0] * (bert_max_input_len - len(self.answer_n_context_ids))
        self.answer_n_context_ids += [0] * (bert_max_input_len - len(self.answer_n_context_ids))
        self.answer_n_context_token_types += [0] * (bert_max_input_len - len(self.answer_n_context_token_types))

        # -- "Tensorizing" --

        self.answer_span_start = torch.Tensor([self.answer_span_start])
        self.answer_span_end = torch.Tensor([self.answer_span_end])

        self.question_n_context_ids = torch.Tensor(self.question_n_context_ids)
        self.question_n_context_mask = torch.Tensor(self.question_n_context_mask)
        self.question_n_context_token_types = torch.Tensor(self.question_n_context_token_types)
        self.question_n_context_context_start_idx = torch.Tensor([self.question_n_context_context_start_idx])  # The index of the first token of the context
        self.question_n_context_context_end_idx = torch.Tensor([self.question_n_context_context_end_idx])  # The index of the final SEP token

        self.answer_n_context_ids = torch.Tensor(self.answer_n_context_ids)
        self.answer_n_context_mask = torch.Tensor(self.answer_n_context_mask)
        self.answer_n_context_token_types = torch.Tensor(self.answer_n_context_token_types)
        self.answer_n_context_context_start_idx = torch.Tensor([self.answer_n_context_context_start_idx])
        self.answer_n_context_context_end_idx = torch.Tensor([self.answer_n_context_context_end_idx])

        self.category = torch.Tensor([self.category])

        answer['input_text'] = answer_text
        answer['discard'] = self.discard
