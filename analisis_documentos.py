import math
import copy
from collections import Counter, OrderedDict
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('spanish'))

df = pd.read_csv('eldiario_grupo_2.csv')


def BoW_vec(docs: list):
    doc_tokens = []
    
   
    for doc in docs:
        tokens = word_tokenize(doc.lower())  
        filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        doc_tokens.append(filtered_tokens)
    
    
    all_tokens = sum(doc_tokens, [])  
    vocab = sorted(set(all_tokens))  
    
   
    zero_vector = OrderedDict((token, 0) for token in vocab)  
    document_bow_vectors = []
    
    for tokens in doc_tokens:
        vec = copy.copy(zero_vector)  
        token_counts = Counter(tokens)  
        for key, count in token_counts.items():
            vec[key] = count
        document_bow_vectors.append(vec)
    
    return document_bow_vectors


document_BoW_vector = BoW_vec(docs=df['Cuerpo'].to_list())


def sim_coseno(vec1, vec2):
    vec1 = [val for val in vec1.values()]
    vec2 = [val for val in vec2.values()]
    dot_prod = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    norm_1 = math.sqrt(sum(x**2 for x in vec1))
    norm_2 = math.sqrt(sum(x**2 for x in vec2))
    return dot_prod / (norm_1 * norm_2) if norm_1 * norm_2 > 0 else 0


num_docs = len(document_BoW_vector)
matriz_semejanza = np.zeros((num_docs, num_docs))

for i in range(num_docs):
    for j in range(num_docs):
        matriz_semejanza[i, j] = sim_coseno(document_BoW_vector[i], document_BoW_vector[j])


representativity_scores = matriz_semejanza.mean(axis=1)


sorted_indices = np.argsort(representativity_scores)[::-1]
most_representative_index = sorted_indices[0]
least_representative_index = sorted_indices[-1]


most_representative_title = df['Titular'].iloc[most_representative_index]
least_representative_title = df['Titular'].iloc[least_representative_index]


print("Documento más representativo:", most_representative_title)
print("Documento menos representativo:", least_representative_title)
print("Puntajes de representatividad:", representativity_scores)

