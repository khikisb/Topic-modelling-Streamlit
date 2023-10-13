import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk import word_tokenize

Data, lda, Model, Implementasi = st.tabs(['Data', 'LDA', 'Modelling', 'Implementasi'])

with Data:
   st.title("UTS Pencarian & Penambangan Web A")
   st.text("Okhi Sahrul Barkah - 210411100112")
   st.subheader("Deskripsi Data")
   st.write("Dimana Fitur yang ada di dalam data tersebut diantaranya:")
   st.text("1) NIM\n2) Judul\n3) Abstrak\n4) Program Studi\n5) Penulis\n6) Dosen Pembimbing 1\n7) Dosen Pembimbing 2\n8) Label")
   st.subheader("Data")
   data = pd.read_csv("DF_PTA.csv")
   st.write(data)

with lda:
   topik = st.number_input("Masukkan Jumlah Topik yang Diinginkan", 1, step=1, value=5)
   lda_model = None  # Inisialisasi lda_model

   def submit():
      tf = pd.read_csv("df_tf.csv")
      lda = LatentDirichletAllocation(n_components=topik, doc_topic_prior=0.2, topic_word_prior=0.1, random_state=42, max_iter=1)
      lda_top = lda.fit_transform(tf)
      # Bobot setiap topik terhadap dokumen
      nama_clm = [f"Topik {i+1}" for i in range(topik)]
      U = pd.DataFrame(lda_top, columns=nama_clm)
      data_with_lda = pd.concat([U, data['Label']], axis=1)
      st.write(data_with_lda)

   all = st.button("Submit")
   if all:
      submit() 

with Model:
    tf = pd.read_csv("df_tf.csv")
    st.subheader("Jumlah Topik yang Anda Gunakan : " + str(topik))
    st.write("Jika pada menu LDA tidak menentukan jumlah topiknya maka proses modelling akan di default dengan jumlah topik = 5")
    lda = LatentDirichletAllocation(n_components=topik, doc_topic_prior=0.2, topic_word_prior=0.1, random_state=42, max_iter=1)
    lda_top = lda.fit_transform(tf)
    # Bobot setiap topik terhadap dokumen
    nama_clm = [f"Topik {i+1}" for i in range(topik)]
    U = pd.DataFrame(lda_top, columns=nama_clm)
    data_with_lda = pd.concat([U, data['Label']], axis=1)
   
    df = data_with_lda.dropna(subset=['Label', 'Label'])

    X = df.drop(columns=['Label']).values
    y = df['Label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model1 = KNeighborsClassifier(5)
    # Pelatihan model KNN dengan data pelatihan
    model1.fit(X_train, y_train)

    model2 = MultinomialNB()
    # Pelatihan model Naive Bayes dengan data pelatihan
    model2.fit(X_train, y_train)

    model3 = DecisionTreeClassifier()
    # Pelatihan model Decision Tree dengan data pelatihan
    model3.fit(X_train, y_train)

    st.write("Pilih metode yang ingin anda gunakan :")
    met1 = st.checkbox("KNN")
    met2 = st.checkbox("Naive Bayes")
    met3 = st.checkbox("Decision Tree")
    submit2 = st.button("Pilih")

    if submit2:      
        if met1:
            st.write("Metode yang Anda gunakan Adalah KNN")
            # Prediksi label kelas pada data pengujian
            y_pred = model1.predict(X_test)
            # Mengukur akurasi model
            accuracy = accuracy_score(y_test, y_pred)
            st.write("Akurasi: {:.2f}%".format(accuracy * 100))
        elif met2:
            st.write("Metode yang Anda gunakan Adalah Naive Bayes")
            # Prediksi label kelas pada data pengujian
            y_pred = model2.predict(X_test)
            # Mengukur akurasi model
            accuracy = accuracy_score(y_test, y_pred)
            st.write("Akurasi: {:.2f}%".format(accuracy * 100))
        elif met3:
            st.write("Metode yang Anda gunakan Adalah Decision Tree")
            # Prediksi label kelas pada data pengujian
            y_pred = model3.predict(X_test)
            # Mengukur akurasi model
            accuracy = accuracy_score(y_test, y_pred)
            st.write("Akurasi: {:.2f}%".format(accuracy * 100))
        else:
            st.write("Anda Belum Memilih Metode")

with Implementasi:
    data = pd.read_csv("DF_PTA.csv")
    data['Abstrak'].fillna("", inplace=True)
    count_vectorizer = CountVectorizer(max_df=0.95, min_df=2)

    st.subheader("Implementasi")
    st.write("Masukkan Abstrak yang Ingin Dianalisis:")
    user_abstract = st.text_area("Abstrak", "")

    def clean_punct(text):
        # Remove Punctuation
        clean_tag = re.compile('@\S+')
        clean_url = re.compile('https?:\/\/.*[\r\n]*')
        clean_hastag = re.compile('#\S+')
        clean_symbol = re.compile('[^a-zA-Z]')
        text = clean_tag.sub('', str(text))
        text = clean_url.sub('', text)
        text = clean_hastag.sub(' ', text)
        text = clean_symbol.sub(' ', text)
        return text

    preprocessing = clean_punct(user_abstract)

    data_clean = []
    for i in range(len(preprocessing)):
        data_clean.append(preprocessing[i])

    tokenize = []
    for i in range(len(data_clean)):
       # Tokenisasi berdasarkan spasi
       token = data_clean[i].split()
       tokenize.append(token)

    stopword = []
    for i in range(len(tokenize)):
        listStopword = set(stopwords.words('indonesian'))
        removed = []
        for x in tokenize[i]:
            if x not in listStopword:
                removed.append(x)
        stopword.append(removed)

    hasilproses = []
    for i in range(len(stopword)):
        joinkata = ' '.join(stopword[i])
        hasilproses.append(joinkata)

    if user_abstract:
        # Fit vocabulary dengan data latih
        count_vectorizer.fit(data['Abstrak'])

        # Transform abstrak pengguna dengan count_vectorizer
        user_tf = count_vectorizer.transform(hasilproses)
        
        if lda_model is None:
            lda_model = LatentDirichletAllocation(n_components=topik, doc_topic_prior=0.2, topic_word_prior=0.1, random_state=42, max_iter=1)
            lda_top = lda_model.fit_transform(user_tf)
            st.write("Model LDA telah dilatih.")

        # Transform abstrak pengguna dengan model LDA
        user_topic_distribution = lda_model.transform(user_tf)
        st.write(user_topic_distribution)
        y_pred = model2.predict(user_topic_distribution)
        y_pred
