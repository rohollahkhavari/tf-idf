# tf-idf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np

corpus  = pd.read_csv('technolife_blog.csv')
docs = corpus.loc [corpus.text.isnull()==bool(False)]['text']

vectorizer = TfidfVectorizer()
tfidf_docs = vectorizer.fit_transform(docs)

query = 'nfc'
tfidf_query = vectorizer.transform([query])[0]

cosines = []

for d in tqdm(tfidf_docs):
    cosines.append(float(cosine_similarity(d, tfidf_query)))   
    
k = 10
sorted_ids = np.argsort(cosines)
for i in range(k):
    cur_id = sorted_ids[-i-1]
    print( cosines[cur_id] ,docs[cur_id])
