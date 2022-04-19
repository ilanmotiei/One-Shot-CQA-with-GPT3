
import matplotlib.pyplot as plt
import argparse


def visualize(logs_file_name):
    logs_file = open(logs_file_name, 'r')

    training_loss = []
    span_start_loss = []
    span_end_loss = []
    type_loss = []

    epoch_training_loss = []
    epoch_span_start_loss = []
    epoch_span_end_loss = []
    epoch_type_loss = []

    test_em = []
    test_f1 = []
    validate_every = None
    gradient_acc_every = 10

    def diff(lst):
        diffs_lst = [((i + 2) * batch_size * a - (i + 1) * batch_size * b) / batch_size for i, (a, b) in enumerate(zip(lst[1:], lst[:-1]))]
        diffs_lst = [lst[0]] + diffs_lst
        return diffs_lst

    lines = logs_file.readlines()

    for i, line in enumerate(lines[1:]):
        if i == 0:
            batch_size = int(line.strip().split()[4].split('/')[0])

        if ' '.join(line.strip().split()[3:5]) == 'Training Start':
            training_loss += diff(epoch_training_loss)
            span_start_loss += diff(epoch_span_start_loss)
            span_end_loss += diff(epoch_span_end_loss)
            type_loss += diff(epoch_type_loss)
            epoch_training_loss = []
            epoch_span_start_loss = []
            epoch_span_end_loss = []
            epoch_type_loss = []

        if line.split()[0] == 'Results:' and validate_every is None:
            validate_every = int(lines[i + 1 - 6].strip().split()[4].split('/')[0])

        if ' '.join(line.split()[:2]) == 'Total Loss:':
            epoch_training_loss.append(float(line.split()[2]))

        elif ' '.join(line.split()[:3]) == 'Span Start Loss:':
            epoch_span_start_loss.append(float(line.split()[3]))

        elif ' '.join(line.split()[:3]) == 'Span End Loss:':
            epoch_span_end_loss.append(float(line.split()[3]))

        elif ' '.join(line.split()[:2]) == 'Type Loss:':
            epoch_type_loss.append(float(line.split()[2]))

        elif line.split()[0].split('(')[0] == 'OrderedDict':
            total_test = lines[i+1].split('(')[-1].split('{')[-1]
            em = float(total_test.split(':')[1].split(',')[0])
            f1 = float(total_test.split(':')[2].split(',')[0])
            test_em.append(em)
            test_f1.append(f1)
            
    plt.plot([batch_size * i / gradient_acc_every for i in range(len(training_loss))], training_loss, label='Training Loss')
    plt.plot([batch_size * i / gradient_acc_every for i in range(len(span_start_loss))], span_start_loss, label='Span Start Loss')
    plt.plot([batch_size * i / gradient_acc_every for i in range(len(span_end_loss))], span_end_loss, label='Span End Loss')
    plt.plot([batch_size * i / gradient_acc_every for i in range(len(span_end_loss))],type_loss, label='Type Loss', linestyle=':')
    plt.xlabel('Step')
    plt.legend()

    plt.show()
    
    plt.plot([validate_every * i / gradient_acc_every for i in range(len(test_em))], test_em, label='Test EM')
    plt.plot([validate_every * i / gradient_acc_every for i in range(len(test_f1))], test_f1, label='Test F1')
    plt.xlabel('Step')
    plt.legend()
    
    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--logs-file-name', dest='logs_file_name', default='logs/logs.txt')
    OPTS = p.parse_args()

    visualize(OPTS.logs_file_name)
