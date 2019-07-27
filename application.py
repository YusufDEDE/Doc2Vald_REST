import os
import re
import json
import nltk
from DocSim import DocSim
from flask_cors import CORS
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from werkzeug.utils import secure_filename
from flask import Flask, jsonify, request, url_for
from gensim.models.keyedvectors import KeyedVectors


UPLOAD_FOLDER ='uploads/'
ALLOWED_EXTENSIONS = set(['txt','pdf'])
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('turkish'))


#Model & Stopword path
Model_Path = 'trmodel'
Stopword_Path ='stopwords-tr.txt'
# text file read and pre-processing begin *>
def read_file(fname):
    with open(fname, 'r') as file: 
        data = file.read()
    return data
def clean_text(text):
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text

# text file read and pre-processing end *>

model = KeyedVectors.load_word2vec_format(Model_Path, binary=True, limit=10 ** 5)

#DocSim script function model & stopword load
with open(Stopword_Path, 'r') as fh:
    stopwords = fh.read().split(",")
ds = DocSim(model,stopwords=stopwords)

#Word2Vec section begin --->
def word_similarity(word):
    return model.similar_by_vector(word)

def paragrah_similarity(source_text, target_text):
    clean_ST = clean_text(source_text)
    clean_TT = clean_text(target_text)
    return ds.calculate_similarity(clean_ST, clean_TT)

#Word2Vec section end --->


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CORS(app)

@app.route('/')
def word_test():
    w = request.args['word']
    return jsonify(word_similarity(w))  

@app.route('/pr', methods=['get','post'])
def paragraph():
    source_text = request.args['source_text']
    target_text = request.args['target_text']
    result = paragrah_similarity(source_text, target_text)
    return json.dumps(str(result))

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        cl = read_file('uploads/'+filename)
        return json.dumps(str(paragrah_similarity(cl, read_file('uploads/clean_text.txt'))))

@app.route('/twoupload', methods=['GET','POST'])
def TwoUploadFile():
    if request.method == 'POST':
        file = request.files['file_one']
        file2 = request.files['file_two']
        filename_one = secure_filename(file.filename)
        filename_two = secure_filename(file2.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename_one))
        file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename_two))
        st = read_file(UPLOAD_FOLDER+filename_one)
        tt = read_file(UPLOAD_FOLDER+filename_two)
        return json.dumps(str(paragrah_similarity(st,tt)))
   
if __name__ == '__main__':
    app.run(debug=True)

