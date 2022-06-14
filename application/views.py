from lib2to3.pgen2 import token
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.core import serializers
from django.http import JsonResponse
import json

''' algo imports'''
import language_tool_python
import pickle 
import re
import nltk, string
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams, pad_sequence, everygrams
from nltk.tokenize import word_tokenize
from nltk.lm import MLE, WittenBellInterpolated
from sklearn.feature_extraction.text import TfidfVectorizer

from . models import Data
# Create your views here.

def home_view(request):
    return render(request, 'application/home.html')

def results_view(request):
    data = Data.objects.get(username='admin')

    essay = data.essay
    grammar_val = data.grammar_val
    mechanics_val = data.mechanics_val
    content_val = data.content_val
    style_val = data.style_val
    plagiarism_val = data.plagiarism_val
    vocabulary_val = data.vocabulary_val

    context = {'essay': essay, 'grammar_val': grammar_val, 'mechanics_val': mechanics_val, 'content_val': content_val, 'style_val': style_val, 'plagiarism_val': plagiarism_val, 'vocabulary_val': vocabulary_val}
    return render(request, 'application/results.html', context)

def check_essay(request):
    topic = request.POST['topic']
    essay = request.POST['essay']

    grammar_val = request.POST['grammarVal']
    mechanics_val = request.POST['mechanicsVal']
    content_val = request.POST['contentVal']
    style_val = request.POST['styleVal']
    plagiarism_val = request.POST['plagiarismVal']
    vocabulary_val = request.POST['vocabularyVal']

    # Grammar
    grammar_matches, grammar_score = grammar(essay)
    # Mechanics
    mechanics_words, mechanics_score = mechanics(essay)
    # Content
    content_words, content_score = content(topic, essay)
    # Style
    style_expressions, style_score = style(essay)
    # Plagiarism
    plagiarised_words, plagiarism_score = plagiarism(essay)
    # Lexical Complexity
    lexical_complexity_words, lexical_complexity_score = lexical_complexity(essay)

    # Update data in user Data
    user = Data.objects.get(username='admin')
    user.topic = topic
    user.essay = essay

    user.grammar_score = grammar_score
    user.grammar_matches = grammar_matches

    user.mechanics_score = mechanics_score
    user.mechanics_words = mechanics_words

    user.content_score = content_score
    user.content_words = content_words

    user.style_score = style_score
    user.style_expressions = style_expressions

    user.plagiarism_score = plagiarism_score
    user.plagiarised_words = plagiarised_words

    user.lexical_complexity_score = lexical_complexity_score
    user.lexical_complexity_words = lexical_complexity_words

    user.grammar_val = float(grammar_val)/100
    user.mechanics_val = float(mechanics_val)/100
    user.content_val = float(content_val)/100
    user.style_val = float(style_val)/100
    user.plagiarism_val = float(plagiarism_val)/100
    user.vocabulary_val = float(vocabulary_val)/100

    user.save()

    response = {'success' : 'success'}
    return JsonResponse(response)

def get_data(request):
    data = Data.objects.get(username='admin')

    data = {'grammar_score': data.grammar_score, 'grammar_matches': data.grammar_matches, 'mechanics_score': data.mechanics_score, 'mechanics_words': data.mechanics_words, 'content_score': data.content_score, 'content_words': data.content_words, 'style_score': data.style_score, 'style_expressions': data.style_expressions, 'plagiarised_words': data.plagiarised_words, 'plagiarism_score': data.plagiarism_score , 'lexical_complexity_score': data.lexical_complexity_score, 'lexical_complexity_words': data.lexical_complexity_words}

    return JsonResponse(data, status=200)




''' GRAMMAR '''
def grammar(essay):
    tool = language_tool_python.LanguageTool('en-US')
    text = essay
    matches = tool.check(text)

    grammar_matches = []
    for match in matches:
        if match.ruleId != 'POSSESSIVE_APOSTROPHE':
            #match_message = match.message.replace('“', "'").replace('”', "'")
            #print(match_message)
            grammar_matches.append({'message': match.message, 'context': match.context, 'replacements': match.replacements})

   # print(matches)
   # print(grammar_matches)
    # Load grammar model
    print('Grammar: ', len(matches))
    grammar = pickle.load(open('./ml/grammar.pkl', "rb"))
    grammar_score = grammar.predict([[len(matches)]])
    grammar_score = round(grammar_score[0])
    return grammar_matches, grammar_score


''' MECHANICS '''
def mechanics(essay):
    char_count = noOfChar(essay)
    word_count = int(noOfWords(essay))
    sent_count = noOfSent(essay)
    avg_word_length = avg_word_len(essay)

    mechanics = pickle.load(open('./ml/mechanics.pkl', "rb"))
    mechanics_score = mechanics.predict([[char_count, word_count, sent_count, avg_word_length]])
    mechanics_score = round(mechanics_score[0])

    if mechanics_score > 100:
        mechanics_score = 100

    print('Char: ', char_count)
    print('Word: ', word_count)
    print('Sent: ', sent_count)
    print('Avg Word Len: ', avg_word_length)

    return word_count, mechanics_score

''' CONTENT '''
def content(topic, essay):
    return cosine_sim(topic, essay)
   

''' STYLE '''
def style(essay):
    noun_count, verb_count, adj_count, adv_count = count_pos(essay) 
    linking_expressions_count, style_expressions = count_linking_expressions(essay)

    style = pickle.load(open('./ml/style.pkl', "rb"))
    style_score = style.predict([[noun_count, verb_count, adj_count, adv_count, linking_expressions_count]])
    style_score = round(style_score[0]*0.88)

    print('Noun: ', noun_count)
    print('Verb: ', verb_count)
    print('Adj: ', adj_count)
    print('Adv: ', adv_count)
    return style_expressions, style_score

''' PLAGIARISM '''
def plagiarism(essay):
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
        #print('i:', i, 'item:', item, 'history: ', testing_data[i:i+n-1], 'Score', s, 'smt: ', testing_data[i:i+n])
            
        scores.append(s)
    
    # Average Plagiarism score
    average = round((sum(scores) / len(scores)) * 100)

    # Reverse Score
    plagiarism_score = average
    #print('Average Plagiarism score: ', plagiarism_score)

    return plagiarised_words, plagiarism_score

''' VOCABULARY '''
def lexical_complexity(essay):
    stopset = set(stopwords.words('english'))
    essay = re.sub("[^A-Za-z0-9]"," ", essay)
    tokens = nltk.word_tokenize(essay)
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if word not in stopset]

    tokens_count = {i: tokens.count(i) for i in tokens}

    words_repeated = []
    for key, value in tokens_count.items():
        if value > 3:
            words_repeated.append(key)

    lexical_complexity = pickle.load(open('./ml/vocabulary.pkl', "rb"))
    lexical_complexity_score = lexical_complexity.predict([[len(words_repeated)]])
    lexical_complexity_score = round(lexical_complexity_score[0])

    print('Words Repeated: ', len(words_repeated))
    return words_repeated, lexical_complexity_score



''' UTILS '''
# Mechanics
def sent2word(x):
    x=re.sub("[^A-Za-z0-9]"," ",x)
    words=nltk.word_tokenize(x)
    return words

def essay2word(essay):
    essay = essay.strip()
    tokenizer = nltk.data.load('./ml/english.pickle')
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

# Style
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

linking_expressions = ['however', 'another', 'moreover', 'and then', 'similarly', 'also', 'in addition', 'likewise', 'as well as', 'besides', 'furthermore', 'besides this', 'in the same way', 'after this', 'then', 'at this point', 'earlier', 'later', 'to begin with', 'initially', 'following this', 'another advantage', 'one reason', 'another reason', 'a further reason', 'eventually', 'so', 'in that case', 'thus', 'consequently', 'thereby', 'therefore', 'as a result', 'admittedly', 'it follows that', 'on the other hand', 'despite', 'in spite of', 'in contrast', 'alternatively', 'although', 'on the contrary', 'instead of', 'rather', 'whereas', 'nonetheless', 'even though', 'obviously', 'certainly', 'plainly', 'undoubtedly', 'since', 'because', 'due to', 'owing to', 'leads to', 'cause of', 'in order to', 'if', 'unless', 'whether', 'provided that', 'depending on', 'in conclusion', 'in summary', 'lastly', 'finally', 'to conclude', 'to recapitulate', 'in short', 'in my opinion', 'to sum up', 'as far as im concerned', 'to my mind', 'it seems to me that']
def count_linking_expressions(essay):
    essay = re.sub("[^A-Za-z0-9]"," ", essay) # Remove puncs
    essay = nltk.word_tokenize(essay)

    count = 0
    linking_expressions_list = []
    for word in essay:
        if word.lower() in linking_expressions:
            count += 1
            linking_expressions_list.append(word.lower())

    print('Linking Exp: ', len(linking_expressions_list))
    return count, linking_expressions_list


# Content
lemmatizer = WordNetLemmatizer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def lemma_tokens(tokens):
    return [lemmatizer.lemmatize(item) for item in tokens]

'''remove punctuation, lowercase, lemmatize'''
def normalize(text):
    return lemma_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')
def cosine_sim(text1, text2):
    heatmaps = vectorizer.fit_transform([text1])
    heatmaps_list = list(vectorizer.vocabulary_.keys())

    tfidf = vectorizer.fit_transform([text1, text2])

    content_score = round((((tfidf * tfidf.T).A)[0,1]*3)*100)
    print('Content Score: ', content_score)
    if content_score > 100:
        content_score = 100
    return heatmaps_list, content_score