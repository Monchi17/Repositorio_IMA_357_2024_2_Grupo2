
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# Cargar el nuevo archivo CSV y seleccionar solo las columnas "Titular" y "Cuerpo"
url = "https://raw.githubusercontent.com/Monchi17/Repositorio_IMA_357_2024_2_Grupo2/main/eldiario_grupo_2.csv"
df = pd.read_csv(url, usecols=['Titular', 'Cuerpo'])

# Título de la aplicación en Streamlit
st.title("App Item 3")

# Mostrar la tabla de documentos
st.write("Tabla de documentos:")
st.dataframe(df[['Titular', 'Cuerpo']])

# Input de palabra para contar la frecuencia
palabra = st.text_input("Introduzca una palabra:")
if palabra:
    # Calcular la frecuencia de la palabra en cada documento
    df['Frecuencia'] = df['Cuerpo'].apply(lambda x: Counter(x.lower().split())[palabra.lower()])
    doc_max_frec = df.loc[df['Frecuencia'].idxmax()]
    
    # Mostrar los resultados para la palabra
    st.write(f"Resultados para '{palabra}':")
    st.write(pd.DataFrame({'Titular': [doc_max_frec['Titular']], 'Frecuencia': [doc_max_frec['Frecuencia']]}))

# Input de oración para similitud coseno
oracion = st.text_input("Ingrese una oración:")
if oracion:
    # Vectorizar los documentos y la oración ingresada usando BoW
    vectorizer = CountVectorizer()
    docs_vectors = vectorizer.fit_transform(df['Cuerpo'].tolist() + [oracion]).toarray()
    doc_vectors = docs_vectors[:-1]
    oracion_vector = docs_vectors[-1]

    # Calcular las similitudes coseno
    similitudes = cosine_similarity([oracion_vector], doc_vectors).flatten()
    
    # Documento con la mayor similitud coseno
    indice_max_similitud = np.argmax(similitudes)
    doc_max_similitud = df.iloc[indice_max_similitud]
    
    # Documento con la mayor suma de frecuencias de tokens de la oración
    oracion_tokens = oracion.lower().split()
    df['Suma_frecuencias'] = df['Cuerpo'].apply(lambda x: sum(Counter(x.lower().split())[token] for token in oracion_tokens))
    doc_max_frecuencia_tokens = df.loc[df['Suma_frecuencias'].idxmax()]

    # Mostrar los resultados
    st.write("Documento con mayor similitud coseno:")
    st.write(pd.DataFrame({'Titular': [doc_max_similitud['Titular']], 'Similitud': [similitudes[indice_max_similitud]]}))

    st.write("Documento con mayor suma de frecuencias de tokens de la oración:")
    st.write(pd.DataFrame({'Titular': [doc_max_frecuencia_tokens['Titular']], 'Suma de Frecuencias': [doc_max_frecuencia_tokens['Suma_frecuencias']]}))
