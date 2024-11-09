
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import nltk
import os

# Configuración de descarga de nltk en una carpeta específica del repositorio
nltk_data_dir = './nltk_data'
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# Descargar stopwords y punkt si no están disponibles
try:
    stop_words = set(nltk.corpus.stopwords.words('spanish'))
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir)
    stop_words = set(nltk.corpus.stopwords.words('spanish'))

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)

# Cargar archivo CSV limpio desde el repositorio de GitHub
url = "https://raw.githubusercontent.com/Monchi17/Repositorio_IMA_357_2024_2_Grupo2/main/resultados_documentos_limpios.csv"
df = pd.read_csv(url)

# Verificar que las columnas requeridas estén en el DataFrame
required_columns = {'Titular', 'Cuerpo_limpio'}
if not required_columns.issubset(df.columns):
    st.error("El archivo CSV no contiene las columnas necesarias: 'Titular' y 'Cuerpo_limpio'")
else:
    # Título de la aplicación en Streamlit
    st.title("App de Análisis de Documentos")

    # Mostrar la tabla de documentos
    st.write("Documentos procesados:")
    st.dataframe(df[['Titular', 'Cuerpo_limpio']])

    # Input de palabra y cálculo de frecuencia
    palabra = st.text_input("Introduzca una palabra para buscar su frecuencia en el documento:")
    if palabra:
        df['Frecuencia'] = df['Cuerpo_limpio'].apply(lambda x: Counter(nltk.word_tokenize(x.lower()))[palabra.lower()])
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
        oracion_tokens = nltk.word_tokenize(oracion.lower())
        df['Suma_frecuencias'] = df['Cuerpo_limpio'].apply(lambda x: sum(Counter(nltk.word_tokenize(x.lower()))[token] for token in oracion_tokens))
        doc_max_frecuencia_tokens = df.loc[df['Suma_frecuencias'].idxmax()]

        # Mostrar resultados
        st.write("Documento con mayor similitud coseno:")
        st.write(pd.DataFrame({'Titular': [doc_max_similitud['Titular']], 'Similitud': [similitudes[indice_max_similitud]]}))

        st.write("Documento con mayor suma de frecuencias de tokens de la oración:")
        st.write(pd.DataFrame({'Titular': [doc_max_frecuencia_tokens['Titular']], 'Suma de Frecuencias': [doc_max_frecuencia_tokens['Suma_frecuencias']]}))
