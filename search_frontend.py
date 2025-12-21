from flask import Flask, request, jsonify
import re
from nltk.corpus import stopwords
from inverted_index_gcp import InvertedIndex
from collections import defaultdict, Counter
import math
import pickle



def build_tokenizer():
    english_stopwords = frozenset(stopwords.words('english'))

    corpus_stopwords = {
        "category", "references", "also", "external", "links",
        "may", "first", "see", "history", "people", "one", "two",
        "part", "thumb", "including", "second", "following",
        "many", "however", "would", "became"
    }

    all_stopwords = english_stopwords.union(corpus_stopwords)

    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

    def tokenize(text):
        tokens = []
        for match in RE_WORD.finditer(text.lower()):
            token = match.group()
            if token not in all_stopwords:
                tokens.append(token)
        return tokens

    return tokenize

def tfidf_cosine_search_body(tokens, body_index, base_dir='body_index'):
    """
    Compute TF-IDF cosine similarity scores for body index.
    Returns a dict: doc_id -> score
    """

    # ---- IDF helper ----
    N = sum(body_index.df.values())  # acceptable corpus size approximation

    def idf(term):
        df = body_index.df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(N / df)

    # ---- build query vector ----
    q_tf = Counter(tokens)
    q_vec = {term: tf * idf(term) for term, tf in q_tf.items()}

    q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
    if q_norm == 0:
        return {}

    # ---- build document vectors ----
    doc_vectors = defaultdict(dict)

    for term, q_weight in q_vec.items():
        posting_list = body_index.read_a_posting_list(
            base_dir=base_dir,
            w=term,
            bucket_name=None
        )
        term_idf = idf(term)

        for doc_id, tf in posting_list:
            doc_vectors[doc_id][term] = tf * term_idf

    # ---- cosine similarity ----
    scores = {}

    for doc_id, d_vec in doc_vectors.items():
        dot = sum(q_vec[t] * d_vec.get(t, 0) for t in q_vec)
        d_norm = math.sqrt(sum(v * v for v in d_vec.values()))
        if d_norm > 0:
            scores[doc_id] = dot / (q_norm * d_norm)

    return scores


def load_body_index():
    """
    Loads the body inverted index from disk / GCP.
    Returns an InvertedIndex object.
    """

    BASE_DIR = 'body_index'          # directory where index files are stored
    INDEX_NAME = 'index'             # index.pkl name without .pkl
    BUCKET_NAME = None               # or 'your-bucket-name' if using GCP

    body_index = InvertedIndex.read_index(
        base_dir=BASE_DIR,
        name=INDEX_NAME,
        bucket_name=BUCKET_NAME
    )

    return body_index
def load_title_index():
    """
    Loads the title inverted index from disk / GCP.
    Returns an InvertedIndex object.
    """
    BASE_DIR = 'title_index'     # folder that contains title index files
    INDEX_NAME = 'index'
    BUCKET_NAME = None

    return InvertedIndex.read_index(
        base_dir=BASE_DIR,
        name=INDEX_NAME,
        bucket_name=BUCKET_NAME
    )
def load_anchor_index():
    BASE_DIR = 'anchor_index'
    INDEX_NAME = 'index'
    BUCKET_NAME = None

    return InvertedIndex.read_index(
        base_dir=BASE_DIR,
        name=INDEX_NAME,
        bucket_name=BUCKET_NAME
    )



def load_doc_titles():
    """
    Loads wiki_id -> title mapping from disk / GCP.
    Returns a dict[int, str].
    """
    TITLES_PATH = "doc_titles.pkl"   # adjust if name/path is different

    with open(TITLES_PATH, "rb") as f:
        doc_titles = pickle.load(f)

    return doc_titles
def score_body(tokens):
    scores = defaultdict(float)
    for term in tokens:
        for doc_id, tf in body_index.read_a_posting_list(
            base_dir='body_index', w=term, bucket_name=None
        ):
            scores[doc_id] += tf
    return scores


def score_title(tokens):
    scores = defaultdict(float)
    for term in set(tokens):
        for doc_id, tf in title_index.read_a_posting_list(
            base_dir='title_index', w=term, bucket_name=None
        ):
            scores[doc_id] += 1
    return scores


def score_anchor(tokens):
    scores = defaultdict(float)
    for term in tokens:
        for doc_id, tf in anchor_index.read_a_posting_list(
            base_dir='anchor_index', w=term, bucket_name=None
        ):
            scores[doc_id] += tf
    return scores

tokenize = build_tokenizer()
body_index = load_body_index()
doc_titles = load_doc_titles()
title_index = load_title_index()
anchor_index = load_anchor_index()
class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    tokens = tokenize(query)
    if not tokens:
        return jsonify(res)

    body_scores = score_body(tokens)
    title_scores = score_title(tokens)
    anchor_scores = score_anchor(tokens)

    final_scores = defaultdict(float)

    for doc_id, score in body_scores.items():
        final_scores[doc_id] += 1.0 * score

    for doc_id, score in title_scores.items():
        final_scores[doc_id] += 2.0 * score

    for doc_id, score in anchor_scores.items():
        final_scores[doc_id] += 1.5 * score

    if not final_scores:
        return jsonify(res)

    ranked_docs = sorted(
        final_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:100]

    res = [
        (doc_id, doc_titles.get(doc_id, ""))
        for doc_id, score in ranked_docs
    ]
    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if (len(query) ==
            0):
      return jsonify(res)
    # BEGIN SOLUTION
    tokens = tokenize(query)
    if not tokens:
        return jsonify(res)
    scores = tfidf_cosine_search_body(tokens, body_index)
    if not scores:
        return jsonify(res)

    ranked_docs = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:100]

    res = [
        (doc_id, doc_titles.get(doc_id, ""))
        for doc_id, score in ranked_docs
        if score > 0
    ]
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    tokens = set(tokenize(query))
    if not tokens:
        return jsonify(res)
    scores = defaultdict(int)
    for term in tokens:
        posting_list = title_index.read_a_posting_list(
            base_dir="title_index",
            w=term,
            bucket_name=None
        )
        for doc_id, tf in posting_list:
            scores[doc_id] += 1

    if not scores:
        return jsonify(res)

    ranked_docs = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    res = [
        (doc_id, doc_titles.get(doc_id, ""))
        for doc_id, score in ranked_docs
    ]
    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    tokens = tokenize(query)
    if not tokens:
        return jsonify(res)

    scores = defaultdict(int)

    for term in tokens:
        posting_list = anchor_index.read_a_posting_list(
            base_dir='anchor_index',
            w=term,
            bucket_name=None
        )
        for doc_id, tf in posting_list:
            scores[doc_id] += tf  # COUNT occurrences

    if not scores:
        return jsonify(res)

    ranked_docs = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    res = [
        (doc_id, doc_titles.get(doc_id, ""))
        for doc_id, score in ranked_docs
    ]
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

def run(**options):
    app.run(**options)

if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
