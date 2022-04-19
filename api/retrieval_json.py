import math
from collections import defaultdict
import regex
import csv
import sys
import time
from transformers import BertTokenizer
import json

import numpy as np

from stopwords import stopwords

csv.field_size_limit(sys.maxsize)


class Indexer:
    remove_terms = stopwords

    def __init__(self, dataset_file):
        self.doc_mat = None
        self.dataset_file = dataset_file
        self.tfidf_by_term = defaultdict(lambda: [])
        self.articles = defaultdict(lambda: '')

    # tokenizer = None
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    @staticmethod
    def tokenize(text):
        terms = regex.sub(r'[\p{P}\p{S}\s]+', ' ', text).lower().strip().split(" ")
        if Indexer.tokenizer is not None:
            tokens = Indexer.tokenizer.encode(text)
            tokens = Indexer.tokenizer.convert_ids_to_tokens(tokens)
            terms = list(filter(lambda t: not regex.fullmatch(r'\[.*\]', t) and regex.match(r'\p{L}', t), tokens))

        return list(filter(lambda s: s not in Indexer.remove_terms, map(lambda s: s.strip(), terms)))

    def precompute(self):
        self.articles = defaultdict(lambda: '')
        all_terms = defaultdict(lambda: [])
        all_tf = dict()

        def processOneTf(id, article):
            self.articles[id] = article

            terms = Indexer.tokenize(article)

            doc_len = len(terms)
            set_terms = set(terms)

            for t in set_terms:
                all_terms[t] += [id]

            all_tf[id] = [(t, terms.count(t) / doc_len) for t in set_terms]

        line = 0
        print("Reading articles...")
        tic = time.perf_counter()
        with open(self.dataset_file, newline='') as f:
            data = json.load(f)['data']

            for challenge in data:
                article = challenge['story']
                qid = challenge['id']

                processOneTf(qid, article)

                line += 1
                if line % 100 == 0:
                    print("  {} processed...".format(line))

        toc = time.perf_counter()
        print("Done, total {} articles, in {:0.2f} minutes".format(line, (toc - tic) / 60))

        print("Generating tfidf...")
        tic = time.perf_counter()

        all_idf = dict()
        all_tfidf = dict()

        document_count = len(all_tf)
        for word, l in all_terms.items():
            all_idf[word] = math.log(document_count / (len(l)))

        line = 0
        for article, terms in all_tf.items():
            all_tfidf[article] = list(map(lambda t: (t[0], float(t[1]) * all_idf[t[0]]), terms))

            line += 1
            if line % 1000 == 0:
                print("  {} processed...".format(line))

        self.tfidf_by_term = defaultdict(lambda: [])
        for word, terms in all_terms.items():
            self.tfidf_by_term[word] = list(map(lambda a: (a, 0), terms))

        for id, terms in all_tfidf.items():
            for saved in terms:
                term = saved[0]
                value = saved[1]

                self.tfidf_by_term[term][self.tfidf_by_term[term].index((id, 0))] = (id, value)

        doc_mat = np.zeros((len(self.articles), len(self.tfidf_by_term)))
        for idx,(t,tfidf) in enumerate(self.tfidf_by_term.items()):
            for doc,value in tfidf:
                doci = list(self.articles.keys()).index(doc)
                doc_mat[doci, idx] = value

        # Normalize matrix document-wise
        self.doc_mat = doc_mat @ np.diag(doc_mat.sum(0)**-1)

        toc = time.perf_counter()
        print("Done, total {} articles in corpus, in {:0.2f} minutes".format(line, (toc - tic) / 60))

    def storecache(self, filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, dialect="excel")

            for term, tfidf in self.tfidf_by_term.items():
                writer.writerow([term] + list(map(lambda t: "{}:{}".format(t[0], t[1]), tfidf)))

    def loadcache(self, cache_filename):
        self.articles = defaultdict(lambda: '')
        with open(self.dataset_file, newline='') as f:
            data = json.load(f)['data']

            for challenge in data:
                article = challenge['story']
                qid = challenge['id']
                self.articles[qid] = article

        with open(cache_filename, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile, dialect="excel")

            self.tfidf_by_term = defaultdict(lambda: [])
            for row in reader:
                self.tfidf_by_term[row[0]] = list(map(lambda r: (r.split(":")[0], float(r.split(":")[1])), row[1:]))



        doc_mat = np.zeros((len(self.articles), len(self.tfidf_by_term)))
        for idx,(t,tfidf) in enumerate(self.tfidf_by_term.items()):
            for doc,value in tfidf:
                doci = list(self.articles.keys()).index(doc)
                doc_mat[doci, idx] = value

        # Normalize matrix document-wise
        self.doc_mat = doc_mat @ np.diag(doc_mat.sum(0)**-1)

    def query_matching_score(self, query, top_n=3):
        terms = Indexer.tokenize(query)

        article_values = defaultdict(lambda: 0)

        for t in terms:
            for doc, value in self.tfidf_by_term[t]:
                article_values[doc] += value

        return list(map(lambda key: (key, article_values[key]),
                        sorted(article_values, key=article_values.get, reverse=True)[:top_n]))

    def query_cosine_similarity(self, query, top_n=3):
        query_terms = Indexer.tokenize(query)

        q_vec = np.array([(query_terms.count(t) / len(query_terms)) * math.log(len(self.articles) / (len(docs)))
                          for t, docs in self.tfidf_by_term.items()])

        norm = np.linalg.norm(q_vec)
        if norm == 0:
            return []
        q_vec = q_vec / norm

        similarity = np.dot(self.doc_mat, q_vec)
        topidx = np.argpartition(similarity, -top_n)[-top_n:]

        return [(list(self.articles.keys())[k], similarity[k]) for k in topidx]



    def getarticle(self, id):
        return self.articles[id]


if __name__ == '__main__':
    dataset_file = 'coqa-dev-v1.0.json'

    inx = Indexer(dataset_file)
    inx.loadcache(dataset_file + ".cache")
    # inx.precompute()
    # inx.storecache(dataset_file + ".cache")

    print()


    def single_question(question):
        tic = time.perf_counter()
        print("---")
        print(question)
        for a, s in inx.query_cosine_similarity(question):
            print("{}: {}".format(a, s))
        toc = time.perf_counter()
        print("Answered in {:0.2f}s".format(toc - tic))
        print()


    single_question("Which is the most powerful nuclear reactor in the world?")
    single_question("Where is Zimbabwe?")
    single_question("Which novels did Nick Cave write?")
