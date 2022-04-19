# Few-Shot-CQA-with-GPT3

Code for the our NLP course paper: 

We investigated the ability of neural models to understand and answer questions with respect to some context, i.e., with respect to previously asked questions and their answers about a given passage. Then we used the learnt model to feed the bigger GPT-3 model (350M parameters) with in-context samples from the training set when queried about a new one.
We propose a method for inferencing with GPT-3 in the conversational question answering task and show that the best results are obtained mostly when using the model that was fine-tuned for that task.

We've also built a retrivel module based on the TD-IDF algorithm together with an API for a convienient use.

For downloading the Repo:

```
git clone 
