from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.core import serializers
from django.http import JsonResponse
import json

import pickle
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import re
from spellchecker import SpellChecker
import language_tool_python

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Create your views here.

def home_view(request):
    return render(request, 'application/home.html')


scores = []
def process_essay(request):
    scores.clear() # Reset value
    # unpack request from front end:
    input_essay = request.POST['essay']


    # Clean Essay
    clean_essay_text = clean_essay(input_essay)

    # Feature Extraction - Mechanics
    char_count = noOfChar(input_essay)
    word_count = noOfWords(input_essay)
    sent_count = noOfSent(input_essay)
    avg_word_len1 = avg_word_len(input_essay)
    spell_err_count1 = spell_err_count(clean_essay_text)

    print('Character count:', char_count)
    print('Word count:', word_count)
    print('Sentence count:', sent_count)
    print('Average word length:', avg_word_len1)
    print('Spelling error count:', spell_err_count1)

    print('--------------------------------')

    # Feature Extraction - Grammar
    noun_count, verb_count, adj_count, adv_count = count_pos(input_essay)
    grammar_err_count1 = grammar_err_count(input_essay)
    
    print('Noun count:', noun_count)
    print('Verb count:', verb_count)
    print('Adjective count:', adj_count)
    print('Adverb count:', adv_count)
    print('Grammar error count:', grammar_err_count1)


    # load xgb_mechanics.pkl
    xgb_mechanics = pickle.load(open('ml/xgb_mechanics.pkl', "rb"))
    # load xgb_grammar.pkl
    xgb_grammar = pickle.load(open('ml/xgb_grammar.pkl', "rb"))

    # Predict Mechanics Score
    prediction_mechanics = xgb_mechanics.predict([char_count, word_count, sent_count, avg_word_len1, spell_err_count1])
    # Predict Grammar Score
    prediction_grammar = xgb_grammar.predict([[noun_count, verb_count, adj_count, adv_count, grammar_err_count1]])

    mechanics_score = int(float(prediction_mechanics[0])*10)
    print("Mechanics score:", mechanics_score)
    grammar_score = int(float(prediction_grammar[0])*10)
    print("Grammar score:", grammar_score)

    # Append to scores list
    scores.append(mechanics_score) 
    scores.append(grammar_score) 

    # Pack response for POST Ajax (required)
    response = {
        'essay' : input_essay,
    }

    return JsonResponse(response)

def get_score(request):
    print(scores)
    response = json.dumps(scores)
    return HttpResponse(response, content_type="application/json")


def remove_puncs(essay): # Remove punctuations
    essay = re.sub("[^A-Za-z ]"," ",essay)
    return essay


def clean_essay(essay):
    essay2 = remove_puncs(essay)
    return essay2



############## Feature Extaction - Mechanics ##################
def sent2word(x):
    x=re.sub("[^A-Za-z0-9]"," ",x)
    words=nltk.word_tokenize(x)
    return words

def essay2word(essay):
    essay = essay.strip()
    tokenizer = nltk.data.load('ml/english.pickle')
    raw = tokenizer.tokenize(essay)
    final_words=[]
    for i in raw:
        if(len(i)>0):
            final_words.append(sent2word(i))
    return final_words
        

def noOfWords(essay):
    count=0
    for i in essay2word(essay):
        count=count+len(i)
    return count

def noOfChar(essay):
    count=0
    for i in essay2word(essay):
        for j in i:
            count=count+len(j)
    return count

def noOfSent(essay):
    return len(essay2word(essay))

def avg_word_len(essay):
    return noOfChar(essay)/noOfWords(essay)

def spell_err_count(essay):
    tokens = word_tokenize(essay)

    spell = SpellChecker()
    
    # find those words that may be misspelled
    misspelled = spell.unknown(tokens)
    if len(misspelled) == 0:
        return 0
    else:
        return len(misspelled)

############# Feature Extraction - Grammar ###########
def count_pos(essay):
    sentences = essay2word(essay)
    noun_count=0
    adj_count=0
    verb_count=0
    adverb_count=0
    for i in sentences:
        pos_sentence = nltk.pos_tag(i)
        for j in pos_sentence:
            pos_tag = j[1]
            if(pos_tag[0]=='N'):
                noun_count+=1
            elif(pos_tag[0]=='V'):
                verb_count+=1
            elif(pos_tag[0]=='J'):
                adj_count+=1
            elif(pos_tag[0]=='R'):
                adverb_count+=1
    return noun_count, verb_count, adj_count, adverb_count

def grammar_err_count(essay):
  tool = language_tool_python.LanguageTool('en-US')
  matches = tool.check(essay)
  return len(matches)