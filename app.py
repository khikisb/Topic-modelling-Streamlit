import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

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
   topik = st.number_input("Masukkan Jumlah Topik yang Diinginkan", 1, step=1)

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
    st.write("Jika pada menu LDA tidak menentukan jumlah topiknya maka proses modelling akan di default dengan jumlah topik = 1")
    lda = LatentDirichletAllocation(n_components=topik, doc_topic_prior=0.2, topic_word_prior=0.1, random_state=42, max_iter=1)
    lda_top = lda.fit_transform(tf)
    data_with_lda = pd.concat([tf, data['Label']], axis=1)
   
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
   from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
   import string
   
   # Preprocessing function
   def preprocess_text(text):
       # Menghilangkan tanda baca
       text = text.translate(str.maketrans('', '', string.punctuation))
   
       # Stopwords removal
       text = ' '.join([word for word in text.lower().split() if word not in ENGLISH_STOP_WORDS])
   
       return text
      
       tf = pd.read_csv("df_tf.csv")
       st.subheader("Jumlah Topik yang Anda Gunakan : " + str(topik))
       st.write("Jika pada menu LDA tidak menentukan jumlah topiknya maka proses modelling akan di default dengan jumlah topik = 1")
       lda = LatentDirichletAllocation(n_components=topik, doc_topic_prior=0.2, topic_word_prior=0.1, random_state=42, max_iter=1)
       lda_top = lda.fit_transform(tf)
       data_with_lda = pd.concat([tf, data['Label']], axis=1)
      
       df = data_with_lda.dropna(subset=['Label', 'Label'])
   
       X = df.drop(columns=['Label']).values
       y = df['Label'].values
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
       # Inisialisasi model KNN
       knn = KNeighborsClassifier(n_neighbors=5)
       knn.fit(X[1:], Label)

   
   # Fungsi untuk memprediksi label
   def predict_label(user_input, lda, knn, vectorizer):
       # Preprocess input abstrak
       user_input_preprocessed = preprocess_text(user_input)
   
       # Mengubah abstrak baru ke dalam vektor topik dengan model LDA
       user_input_vector = lda.transform(vectorizer.transform([user_input_preprocessed]))
   
       # Memprediksi label untuk input pengguna
       predicted_label = knn.predict(user_input_vector)
       return predicted_label[0]
   
   # Contoh penggunaan:
   user_input = "Abstrak baru yang ingin diprediksi"
   predicted_label = predict_label(user_input, lda, knn, vectorizer)
   print(f"Label yang diprediksi: {predicted_label}")


