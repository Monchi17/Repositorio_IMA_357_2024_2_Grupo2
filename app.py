import streamlit as st
import pandas as pd
import numpy as np
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import nltk
import os

# Configurar la descarga de recursos de NLTK en un directorio específico dentro del repositorio
nltk_data_dir = './nltk_data'
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# Descargar stopwords y punkt si no están disponibles
try:
    stop_words = set(stopwords.words('spanish'))
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir)
    stop_words = set(stopwords.words('spanish'))

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)
    
# Cargar archivo CSV desde el repositorio de GitHub
url = "https://raw.githubusercontent.com/Monchi17/Repositorio_IMA_357_2024_2_Grupo2/main/eldiario_grupo_2.csv"
df = pd.read_csv(url)


st.title("App Item 3")
st.write("Tabla de documentos:")
st.dataframe(df)


stop_words = set(stopwords.words('spanish'))


palabra = st.text_input("Introduzca una palabra:")
if palabra:
    
    df['Frecuencia'] = df['Cuerpo'].apply(lambda x: Counter(word_tokenize(x.lower()))[palabra])
    doc_max_frec = df.loc[df['Frecuencia'].idxmax()]
    
    
    st.write(f"Resultados para '{palabra}':")
    st.write(pd.DataFrame({'Titular': [doc_max_frec['Titular']], 'Frecuencia': [doc_max_frec['Frecuencia']]}))

oracion = st.text_input("Ingrese una oración:")
if oracion:
    oracion_tokens = [word for word in word_tokenize(oracion.lower()) if word.isalpha() and word not in stop_words]
    
    
    def suma_frecuencias(doc):
        doc_tokens = word_tokenize(doc.lower())
        doc_counter = Counter(doc_tokens)
        return sum(doc_counter[token] for token in oracion_tokens)
    
    df['Suma_Frecuencias'] = df['Cuerpo'].apply(suma_frecuencias)
    doc_max_suma = df.loc[df['Suma_Frecuencias'].idxmax()]


    vectorizer = CountVectorizer().fit_transform(df['Cuerpo'].tolist() + [oracion])
    vectors = vectorizer.toarray()
    cos_similarities = cosine_similarity([vectors[-1]], vectors[:-1])[0]


    doc_max_cosine = df.iloc[cos_similarities.argmax()]
    max_cosine_sim = cos_similarities.max()

    # Mostrar resultados
    st.write("El titular del cuerpo más similar por similitud coseno es:")
    st.write(f"* **Título**: {doc_max_cosine['Titular']}")
    st.write(f"* **Similitud**: {max_cosine_sim:.4f}")

    st.write("El titular del cuerpo con la mayor suma de frecuencias es:")
    st.write(f"* **Título**: {doc_max_suma['Titular']}")
    st.write(f"* **Mayor coincidencia**: {doc_max_suma['Suma_Frecuencias']} palabras encontradas")
