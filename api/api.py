import webbrowser

from flask import Flask
from flask import request, render_template
import torch
import sys
import os
cqa_path = os.path.realpath('.')
sys.path.insert(1, cqa_path)
from cqa.model import CQAer
import argparse
from retrieval_json import Indexer

app = Flask(__name__)


@app.route("/")
def index():
    return render_template('website.html')


def question_answer(model, question, text, state):
    state, answer, confidence_score = model.answer1(context=text, question=question, state=state)

    return answer, state, confidence_score


@app.route("/answer", methods=['POST'])
def answer():

    question = request.form.get('question')
    use_retrieval = request.form.get('retrieval') == 'true'
    context = request.form.get('context')
    resetHistory = request.form.get('reset') == 'true'

    if resetHistory:
        vars_dict['state'] = None
        vars_dict['context'] = context

    context = vars_dict['context']

    if use_retrieval and resetHistory:
        max_confidence = -1
        max_confidence_article = ("", "---", "?")
        for id, _ in inx.query_cosine_similarity(question, top_n=3):
            article = inx.getarticle(id)
            answer, _, confidence = question_answer(model, question=question, text=article, state=vars_dict['state'])
            if confidence >= max_confidence:
                max_confidence = confidence
                max_confidence_article = (id, article, answer)

        vars_dict['context'] = max_confidence_article[1]
        return {"answer": max_confidence_article[2], "context": max_confidence_article[1], "question_id": max_confidence_article[0]}
    else:
        answer, vars_dict['state'], _ = question_answer(model, question=question, text=context, state=vars_dict['state'])
        return {"answer": answer, "context": context, "question_id": ""}


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Parser for the retreival api')

    parser.add_argument('--args', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()

    vars_dict = {'context': None, 'state': None}
    model_args = torch.load(args.args, map_location='cpu')
    model_args.device = 'cpu'
    model = CQAer(model_args).to(model_args.device)
    model.eval()
    model.load_state_dict(torch.load(args.model, map_location='cpu'))
    # model.to('cpu')

    dataset_file = args.data

    inx = Indexer(dataset_file)
    inx.loadcache(dataset_file + ".cache")
    # inx.precompute()
    # inx.storecache(dataset_file + ".cache")

    webbrowser.open_new('http://127.0.0.1:2000/')

    app.run(port=2000)
