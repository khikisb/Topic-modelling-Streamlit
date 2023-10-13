import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

Data, lda, Model = st.tabs(['Data', 'LDA', 'Modelling'])

with Data:
   st.title("UTS Pencarian & Penambangan Web A")
   st.text("Okhi Sahrul Barkah")
   st.subheader("Deskripsi Data")
   st.write("Dimana Fitur yang ada di dalam data tersebut diantaranya:")
   st.text("1) Judul\n2) Penulis\n3) Dosen Pembimbing 1\n4) Dosen Pembinbing 2\n5) Abstrak\n6) Label")
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
        data_with_lda = pd.concat([U, data], axis=1)
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
    y = data.Label
    X_train, X_test, y_train, y_test = train_test_split(lda_top, y, test_size=0.2, random_state=42)
    
    metode1 = KNeighborsClassifier(n_neighbors=3)
    metode1.fit(X_train, y_train)

    metode2 = GaussianNB()
    metode2.fit(X_train, y_train)

    metode3 = tree.DecisionTreeClassifier(criterion="gini")
    metode3.fit(X_train, y_train)

    st.write("Pilih metode yang ingin anda gunakan :")
    met1 = st.checkbox("KNN")
    # if met1 :
    #     st.write("Hasil Akurasi Data Training Menggunakan KNN sebesar : ", (100 * metode1.score(X_train, y_train)))
    #     st.write("Hasil Akurasi Data Testing Menggunakan KNN sebesar : ", (100 * (metode1.score(X_test, y_test))))
    met2 = st.checkbox("Naive Bayes")
    # if met2 :
    #     st.write("Hasil Akurasi Data Training Menggunakan Naive Bayes sebesar : ", (100 * metode2.score(X_train, y_train)))
    #     st.write("Hasil Akurasi Data Testing Menggunakan Naive Bayes sebesar : ", (100 * metode2.score(X_test, y_test)))
    met3 = st.checkbox("Decision Tree")
    # if met3 :
        # st.write("Hasil Akurasi Data Training Menggunakan Decision Tree sebesar : ", (100 * metode3.score(X_train, y_train)))
        # st.write("Hasil Akurasi Data Testing Menggunakan Decision Tree sebesar : ", (100 * metode3.score(X_test, y_test)))
    submit2 = st.button("Pilih")

    if submit2:      
        if met1:
            st.write("Metode yang Anda gunakan Adalah KNN")
            st.write("Hasil Akurasi Data Training Menggunakan KNN sebesar : ", (100 * metode1.score(X_train, y_train)))
            st.write("Hasil Akurasi Data Testing Menggunakan KNN sebesar : ", (100 * metode1.score(X_test, y_test)))
        elif met2:
            st.write("Metode yang Anda gunakan Adalah Naive Bayes")
            st.write("Hasil Akurasi Data Training Menggunakan Naive Bayes sebesar : ", (100 * metode2.score(X_train, y_train)))
            st.write("Hasil Akurasi Data Testing Menggunakan Naive Bayes sebesar : ", (100 * metode2.score(X_test, y_test)))
        elif met3:
            st.write("Metode yang Anda gunakan Adalah Decision Tree")
            st.write("Hasil Akurasi Data Training Menggunakan Decision Tree sebesar : ", (100 * metode3.score(X_train, y_train)))
            st.write("Hasil Akurasi Data Testing Menggunakan Decision Tree sebesar : ", (100 * metode3.score(X_test, y_test)))
        else:
            st.write("Anda Belum Memilih Metode")
    # else:
    #     st.write("Anda Belum Menentukan Jumlah Topik di Menu LDA")
