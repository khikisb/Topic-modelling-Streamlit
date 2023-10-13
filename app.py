import streamlit as st
import pandas as pd
import numpy as np
from numpy import array
import pickle
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

Data, Ekstraksi, lda, Model = st.tabs(['Data', 'Ekstraksi Fitur', 'LDA', 'Modelling'])

with Data :
   st.title("""UTS Pencarian & Penambangan Web A""")
   st.text('Zuni Amanda Dewi 200411100051')
   st.subheader('Deskripsi Data')
   st.write("""Dimana Fitur yang ada di dalam data tersebut diantaranya :""")
   st.text("""
            1) Judul
            2) Penulis
            3) Dosen Pembimbing 1
            4) Dosen Pembinbing 2
            5) Abstrak
            5) Label""")
   st.subheader('Data')
   data=pd.read_csv('crawling_pta_labeled.csv')
   data

with Ekstraksi :
   url_tf='https://drive.google.com/file/d/1bmViR9avCJYNdVjgrKCKS7vl2AOu6W8A/view?usp=sharing'
   url_log_tf='https://drive.google.com/file/d/1-0mBse0FBN9bLUZU8cG4iMpLDW2HolS7/view?usp=sharing'
   url_oht='https://drive.google.com/file/d/1-4qqy-4kBvZ_k_BBxTetiZrA0Aj8zIyX/view?usp=sharing'
   url_tf_idf='https://drive.google.com/file/d/1-5bke07KeJ3oiF5Mt0jicQVFgVBRSHUq/view?usp=sharing'
   file_id1=url_tf.split('/')[-2]
   file_id2=url_log_tf.split('/')[-2]
   file_id3=url_oht.split('/')[-2]
   file_id4=url_tf_idf.split('/')[-2]

   st.subheader('Term Frequency (TF)')
   dwn_url1='https://drive.google.com/uc?id=' + file_id1
   tf = pd.read_csv(dwn_url1)
   tf
   
   st.subheader('Logarithm Frequency (Log-TF)')
   dwn_url2='https://drive.google.com/uc?id=' + file_id2
   log_tf = pd.read_csv(dwn_url2)
   log_tf
   
   st.subheader('One Hot Encoder / Binary')
   dwn_url3='https://drive.google.com/uc?id=' + file_id3
   oht = pd.read_csv(dwn_url3)
   oht
   
   st.subheader('TF-IDF')
   dwn_url4='https://drive.google.com/uc?id=' + file_id4
   tf_idf = pd.read_csv(dwn_url4)
   tf_idf

with lda:
   topik = st.number_input("Masukkan Jumlah Topik yang Diinginkan", 1, step=1)

   def submit():
        lda = LatentDirichletAllocation(n_components=topik, doc_topic_prior=0.2, topic_word_prior=0.1,random_state=42,max_iter=1)
        x=tf.drop('Label', axis=1)
        lda_top=lda.fit_transform(x)
        #bobot setiap topik terhadap dokumen
        nama_clm =[]
        for i in range(topik):
            nama_clm.append(("Topik "+ str(i+1)))
        U = pd.DataFrame(lda_top, columns=nama_clm)
        U['Label']=tf['Label'].values
        U
   all = st.button("Submit")
   if all :
      st.balloons()
      submit()

with Model :
    # if all :
        st.subheader('Jumlah Topik yang Anda Gunakan : ' +str(topik))
        st.write ("Jika pada menu LDA tidak menentukan jumlah topiknya maka proses modelling akan di default dengan jumlah topik = 1")
        lda = LatentDirichletAllocation(n_components=topik, doc_topic_prior=0.2, topic_word_prior=0.1,random_state=42,max_iter=1)
        x=tf.drop(columns='Label')
        lda_top=lda.fit_transform(x)
        y = tf.Label
        X_train,X_test,y_train,y_test = train_test_split(lda_top,y,test_size=0.2,random_state=42)
        
        metode1 = KNeighborsClassifier(n_neighbors=3)
        metode1.fit(X_train, y_train)

        metode2 = GaussianNB()
        metode2.fit(X_train, y_train)

        metode3 = tree.DecisionTreeClassifier(criterion="gini")
        metode3.fit(X_train, y_train)

        st.write ("Pilih metode yang ingin anda gunakan :")
        met1 = st.checkbox("KNN")
        # if met1 :
        #     st.write("Hasil Akurasi Data Training Menggunakan KNN sebesar : ", (100 * metode1.score(X_train, y_train)))
        #     st.write("Hasil Akurasi Data Testing Menggunakan KNN sebesar : ", (100 * (metode1.score(X_test, y_test))))
        met2 = st.checkbox("Naive Bayes")
        # if met2 :
        #     st.write("Hasil Akurasi Data Training Menggunakan Naive Bayes sebesar : ", (100 * metode2.score(X_train, y_train)))
        #     st.write("Hasil Akurasi Data Testing Menggunakan Naive Bayes sebesar : ", (100 * metode2.score(X_test, y_test)))
        met3 = st.checkbox("Decesion Tree")
        # if met3 :
            # st.write("Hasil Akurasi Data Training Menggunakan Decission Tree sebesar : ", (100 * metode3.score(X_train, y_train)))
            # st.write("Hasil Akurasi Data Testing Menggunakan Decission Tree sebesar : ", (100 * metode3.score(X_test, y_test)))
        submit2 = st.button("Pilih")

        if submit2:      
            if met1 :
                st.write("Metode yang Anda gunakan Adalah KNN")
                st.write("Hasil Akurasi Data Training Menggunakan KNN sebesar : ", (100 * metode1.score(X_train, y_train)))
                st.write("Hasil Akurasi Data Testing Menggunakan KNN sebesar : ", (100 * (metode1.score(X_test, y_test))))
            elif met2 :
                st.write("Metode yang Anda gunakan Adalah Naive Bayes")
                st.write("Hasil Akurasi Data Training Menggunakan Naive Bayes sebesar : ", (100 * metode2.score(X_train, y_train)))
                st.write("Hasil Akurasi Data Testing Menggunakan Naive Bayes sebesar : ", (100 * metode2.score(X_test, y_test)))
            elif met3 :
                st.write("Metode yang Anda gunakan Adalah Decesion Tree")
                st.write("Hasil Akurasi Data Training Menggunakan Decission Tree sebesar : ", (100 * metode3.score(X_train, y_train)))
                st.write("Hasil Akurasi Data Testing Menggunakan Decission Tree sebesar : ", (100 * metode3.score(X_test, y_test)))
            else :
                st.write("Anda Belum Memilih Metode")
    # else:
    #     st.write("Anda Belum Menentukan Jumlah Topik di Menu LDA")
