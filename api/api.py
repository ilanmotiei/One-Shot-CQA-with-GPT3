import webbrowser

from flask import Flask
from flask import request, render_template
import torch
from model import CQAer

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
    dataset_file = 'coqa-dev-v1.0.json'

    inx = Indexer(dataset_file)
    inx.loadcache(dataset_file + ".cache")
    # inx.precompute()
    # inx.storecache(dataset_file + ".cache")



    vars_dict = {'context': None, 'state': None}
    args = torch.load('args_saved.pth', map_location='cpu')
    args.device = 'cpu'
    model = CQAer(args).to(args.device)
    model.eval()
    model.load_state_dict(torch.load('model_epoch=3_saved.pth', map_location='cpu'))
    # model.to('cpu')

    webbrowser.open_new('http://127.0.0.1:2000/')

    app.run(port=2000)
