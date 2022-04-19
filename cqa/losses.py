
import torch


def padded_passages_logits_loss(logits, answer_start_or_end_idx, criterion):
    """
    :param logits: shape = (batch, L, MAX_i_(len(P_i))) where P_i is the length of the context of the i'th challenge
                    at the batch.
    :param answer_start_or_end_idx: shape = (batch, L). The index in which the answer starts/ends indexes at the
            encoded context_ids tensor.
    :param criterion: The criterion for measuring the loss.
    :return: The calculated loss.
    """

    return criterion(logits.transpose(dim0=1, dim1=2), answer_start_or_end_idx)


def in_span_loss_with_logits(in_span_logits, answer_span_start, answer_span_end, lengths, criterion, device):
    # in_span_logits.shape = (N, L, max_passage_len)
    # answer_span_start.shape == answer_span_end.shape == (N, L)
    # lengths - a list of size N containing the lengths of the passages at the batch
    # criterion - always be BCEwithLogits(reduction='none') !

    N, L, max_passage_len = in_span_logits.shape

    in_span_targets = torch.stack([torch.stack([torch.cat([torch.zeros(size=(max(0, answer_span_start[b][l]),)),
                                                           torch.ones(size=(answer_span_end[b][l] - answer_span_start[b][l],)),
                                                           torch.zeros(size=(min(max_passage_len - answer_span_end[b][l], max_passage_len),))])
                                                for l in range(L)])
                                   for b in range(N)])

    in_span_loss_mask = torch.stack(
        [torch.cat([torch.ones(size=(L, int(lengths[b].item()))), torch.zeros(size=(L, max_passage_len - int(lengths[b].item())))],
                   dim=1)
         for b in range(N)])

    in_span_targets = in_span_targets.to(device)
    in_span_loss_mask = in_span_loss_mask.to(device)

    in_span_loss = torch.sum(criterion(in_span_logits, in_span_targets) * in_span_loss_mask) / \
                   torch.sum(in_span_loss_mask)

    return in_span_loss

