import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re

# Cargar archivo CSV limpio desde el repositorio de GitHub
url = "https://raw.githubusercontent.com/Monchi17/Repositorio_IMA_357_2024_2_Grupo2/main/resultados_documentos_limpios.csv"
df = pd.read_csv(url)

# Título de la aplicación en Streamlit
st.title("App de Análisis de Documentos")

# Mostrar la tabla de documentos
st.write("Documentos procesados:")
st.dataframe(df[['Titular', 'Cuerpo_limpio']])

# Función para tokenizar sin nltk
def tokenizar(texto):
    return [palabra.lower() for palabra in re.findall(r'\b\w+\b', texto) if palabra.isalpha()]

# Input de palabra y cálculo de frecuencia
palabra = st.text_input("Introduzca una palabra para buscar su frecuencia en el documento:")
if palabra:
    df['Frecuencia'] = df['Cuerpo_limpio'].apply(lambda x: Counter(tokenizar(x))[palabra.lower()])
    doc_max_frec = df.loc[df['Frecuencia'].idxmax()]
    
    st.write(f"Resultados para la palabra '{palabra}':")
    st.write(pd.DataFrame({'Titular': [doc_max_frec['Titular']], 'Frecuencia': [doc_max_frec['Frecuencia']]}))

# Input de oración y cálculo de similitud coseno
oracion = st.text_input("Ingrese una oración para comparar similitud:")
if oracion:
    # Vectorizar documentos y oración ingresada usando BoW
    vectorizer = CountVectorizer()
    docs_vectors = vectorizer.fit_transform(df['Cuerpo_limpio'].tolist() + [oracion]).toarray()
    doc_vectors = docs_vectors[:-1]  # Vectores de los documentos
    oracion_vector = docs_vectors[-1]  # Vector de la oración ingresada

    # Calcular similitudes coseno
    similitudes = cosine_similarity([oracion_vector], doc_vectors).flatten()
    
    # Documento con mayor similitud
    indice_max_similitud = np.argmax(similitudes)
    doc_max_similitud = df.iloc[indice_max_similitud]
    
    # Documento con mayor suma de frecuencias de tokens de la oración
    oracion_tokens = tokenizar(oracion)
    df['Suma_frecuencias'] = df['Cuerpo_limpio'].apply(lambda x: sum(Counter(tokenizar(x))[token] for token in oracion_tokens))
    doc_max_frecuencia_tokens = df.loc[df['Suma_frecuencias'].idxmax()]

    # Mostrar resultados
    st.write("Documento con mayor similitud coseno:")
    st.write(pd.DataFrame({'Titular': [doc_max_similitud['Titular']], 'Similitud': [similitudes[indice_max_similitud]]}))

    st.write("Documento con mayor suma de frecuencias de tokens de la oración:")
    st.write(pd.DataFrame({'Titular': [doc_max_frecuencia_tokens['Titular']], 'Suma de Frecuencias': [doc_max_frecuencia_tokens['Suma_frecuencias']]}))
