from flask import Flask, request, render_template


# Extractive summarizer libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Abstractive Summarizer libraries
from transformers import pipeline, BertTokenizer

# Topic Modelling libraries
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')


# WSGI server
app = Flask(__name__)


# Extractive summarizer
def extractiveTokenizer(text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text,])
    return tfidf_matrix

def extractiveSummarizer(text, tfidf_matrix):
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    sentence_scores = cosine_sim[0]
    top_sentence = max(range(len(sentence_scores)), key=sentence_scores.__getitem__)
    return text.split('.')[top_sentence]



# Abstractive Summarizer
def abstractiveTokenizer(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)
    return tokens

def abstractiveSummarizer(text):
    model_name = "sshleifer/distilbart-cnn-12-6"
    summarizer = pipeline('summarization', model=model_name, tokenizer=model_name)
    summary = summarizer(text, max_length=20, min_length=10, length_penalty=2.0, num_beams=4, early_stopping=True)
    return summary[0]['summary_text']


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/exs', methods=['POST'])
def exs():
    return render_template('extractiveForm.html')

@app.route('/abs', methods=['POST'])
def indexxx():
    return render_template('abstractiveForm.html')

@app.route('/topic', methods=['POST'])
def indexxxx():
    return render_template('topicModellingForm.html')

@app.route('/extractiveSummarize', methods=['POST'])
def extractiveSummarize():
    text = request.form['text']
    tokens = extractiveTokenizer(text)
    summary = extractiveSummarizer(text, tokens)
    return render_template('summary.html', text=text, summary=summary)

@app.route('/abstractiveSummarize', methods=['POST'])
def abstractiveSummarize():
    text = request.form['text']
    tokens = abstractiveTokenizer(text)
    summary = abstractiveSummarizer(text)
    return render_template('summary.html', text=text, summary=summary)

@app.route('/topicModellingSummarize', methods=['POST'])
def topicModellingSummarize():
    text = request.form['text']

    # Tokenize and remove stopwords
    tokens = [word for word in word_tokenize(text.lower()) if word.isalpha() and word not in stopwords.words('english')]

    # Create a dictionary and a corpus
    dictionary = corpora.Dictionary([tokens])
    corpus = [dictionary.doc2bow(tokens)]

    # Build the LDA model
    lda_model = models.LdaModel(corpus, num_topics=1, id2word=dictionary, passes=10)

        # Extract words within double quotes using regex
    matches = re.findall(r'"([^"]*)"', lda_model.print_topics()[0][1])

    # Concatenate the words separated by commas
    result_string = ', '.join(matches)
    return render_template('summary.html', text=text, summary=result_string)

if __name__ == '__main__':
    app.run(debug=True)