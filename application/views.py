from lib2to3.pgen2 import token
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.core import serializers
from django.http import JsonResponse
import json

from . models import Data
# Create your views here.

def home_view(request):
    return render(request, 'application/home.html')

def results_view(request):
    data = Data.objects.get(username='admin')

    essay = data.essay
    context = {'essay': essay}
    return render(request, 'application/results.html', context)

def check_essay(request):
    essay = request.POST['essay']

    plagiarised_words, plagiarism_score = plagiarism(essay)
    #print('Plagiarised words: ', plagiarised_words)
    print('Plagiarism Score: ', plagiarism_score)
    
    # update data in user Data
    user = Data.objects.get(username='admin')
    user.plagiarised_words = plagiarised_words
    user.plagiarism_score = plagiarism_score
    user.essay = essay
    user.save()

    response = {'success' : 'success'}
    return JsonResponse(response)

def get_data(request):
    data = Data.objects.get(username='admin')

    print(data.plagiarised_words)
    data = {'plagiarised_words': data.plagiarised_words, 'plagiarism_score': data.plagiarism_score}
   # if data:
   #     response = json.loads(data)
   # else:
   #     response = []

    return JsonResponse(data, status=200)



def plagiarism(essay):
    import re
    import nltk
    nltk.download('punkt')
    from nltk.util import ngrams, pad_sequence, everygrams
    from nltk.tokenize import word_tokenize
    from nltk.lm import MLE, WittenBellInterpolated
    import numpy as np

    # Training data file
    train_data_file = "./ml/plagiarism/train_data.txt"

    # read training data
    with open(train_data_file) as f:
        train_text = f.read().lower()

    # apply preprocessing (remove text inside square and curly brackets and rem punc)
    train_text = re.sub(r"\[.*\]|\{.*\}", "", train_text)
    train_text = re.sub(r'[^\w\s]', "", train_text)

    # set ngram number
    n = 4

    # pad the text and tokenize
    training_data = list(pad_sequence(word_tokenize(train_text), n, pad_left=True, left_pad_symbol="<s>"))
    
    # generate ngrams
    ngrams = list(everygrams(training_data, max_len=n))

    # build ngram language models
    model = WittenBellInterpolated(n)
    model.fit([ngrams], vocabulary_text=training_data)

    # Read testing data
    test_text = essay.lower()
    test_text = re.sub(r'[^\w\s]', "", test_text)

    # Tokenize and pad the text
    testing_data = list(pad_sequence(word_tokenize(test_text), n, pad_left=True, left_pad_symbol="<s>"))
   
    # assign scores
    scores = []
    plagiarised_words = []
    for i, item in enumerate(testing_data[n-1:]):
        s = model.score(item, testing_data[i:i+n-1])
        if s > 0.4:
            s = 1.0
            plagiarised_words.append(item)
        print('i:', i, 'item:', item, 'history: ', testing_data[i:i+n-1], 'Score', s, 'smt: ', testing_data[i:i+n])
            
        scores.append(s)
    
    # Average Plagiarism score
    average = round((sum(scores) / len(scores)) * 100)

    # Reverse Score
    plagiarism_score = average
    #print('Average Plagiarism score: ', plagiarism_score)

    return plagiarised_words, plagiarism_score
