from collections import OrderedDict
import matplotlib.pyplot as plt


bert_finetuned_dicts = [line.strip().split('%')[0].split('----------')[-1].strip() for line in open('logs/bert_finetuned_logs_k=[0,16].txt', 'r').readlines()]
bert_finetuned_dicts = [eval(a) for a in bert_finetuned_dicts if 'OrderedDict' in a]

bert_not_finetuned_dicts = [line.strip().split('%')[0].split('----------')[-1].strip() for line in open('logs/bert_not_finetuned_k=[0,15]_logs.txt', 'r').readlines()]
bert_not_finetuned_dicts = [eval(a) for a in bert_not_finetuned_dicts if 'OrderedDict' in a]

random_same_source_dicts = [eval(line.strip().split('%')[0]) for line in open('logs/same_source_random_logs.txt', 'r').readlines()]
totally_random_dicts = [eval(line.strip().split('%')[0]) for line in open('logs/totally_random_logs_times_[1,5].txt', 'r').readlines()]


def plot(domain, metric, show=True):
    metric_finetuned_bert = [dict[domain][metric] for dict in bert_finetuned_dicts][:-1]
    metric_not_finetuned_bert = [dict[domain][metric] for dict in bert_not_finetuned_dicts]
    metric_random_same_source = [dict[domain][metric] for dict in random_same_source_dicts]
    metric_totally_random = [dict[domain][metric] for dict in totally_random_dicts]

    avg_metric_random_same_source = sum(metric_random_same_source) / len(metric_random_same_source)
    avg_metric_totally_random = sum(metric_totally_random) / len(metric_totally_random)

    ks = list(range(16))

    plt.plot(ks, metric_finetuned_bert, label='Fine-tuned BERT')
    plt.plot(ks, metric_not_finetuned_bert, label='Not Fine-tuned BERT')
    plt.plot(ks, [avg_metric_random_same_source] * len(ks), label='Random Same Source - AVG')
    plt.plot(ks, [avg_metric_totally_random] * len(ks), label='Totally Random - AVG')

    plt.title(f'{domain}: {metric}')
    plt.xlabel('K')

    plt.legend()

    if show:
        plt.show()
    else:
        plt.savefig(f"logs/figs/{domain}:{metric}.png", dpi=1000)
        plt.clf()


if __name__ == "__main__":
    domains = ['children_stories', 'literature', 'mid-high_school', 'news', 'wikipedia', 'overall']
    
    for domain in domains:
        plot(domain, metric='f1', show=False)
        plot(domain, metric='em', show=False)
