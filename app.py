import streamlit as st
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

st.title("UTS Pencarian & Penambangan Web A")
st.text("Okhi Sahrul Barkah")

Data, lda, Model = st.beta_columns(3)

with Data:
    st.subheader("Deskripsi Data")
    st.write("Dimana Fitur yang ada di dalam data tersebut diantaranya :")
    st.text("1) Judul\n2) Penulis\n3) Dosen Pembimbing 1\n4) Dosen Pembinbing 2\n5) Abstrak\n6) Label")
    st.subheader("Data")
    data = pd.read_csv("DF_PTA_LABEL.csv")
    st.write(data)

with lda:
    topik = st.number_input("Masukkan Jumlah Topik yang Diinginkan", 1, step=1)

    def submit():
        df_tf = pd.read_csv("TF_label.csv")
        lda = LatentDirichletAllocation(n_components=topik, doc_topic_prior=0.2, topic_word_prior=0.1, random_state=42, max_iter=1)
        x = df_tf.drop("Label", axis=1)
        lda_top = lda.fit_transform(x)
        nama_clm = [f"Topik {i+1}" for i in range(topik)]
        U = pd.DataFrame(lda_top, columns=nama_clm)
        U["Label"] = df_tf["Label"].values
        st.write(U)

    if st.button("Submit"):
        st.balloons()
        submit()

with Model:
    st.subheader("Jumlah Topik yang Anda Gunakan : " + str(topik))
    st.write("Jika pada menu LDA tidak menentukan jumlah topiknya maka proses modelling akan di default dengan jumlah topik = 1")
    lda = LatentDirichletAllocation(n_components=topik, doc_topic_prior=0.2, topic_word_prior=0.1, random_state=42, max_iter=1)
    x = df_tf.drop(columns="Label")
    lda_top = lda.fit_transform(x)
    y = df_tf["Label"]
    X_train, X_test, y_train, y_test = train_test_split(lda_top, y, test_size=0.2, random_state=42)

    metode1 = KNeighborsClassifier(n_neighbors=3)
    metode1.fit(X_train, y_train)

    metode2 = GaussianNB()
    metode2.fit(X_train, y_train)

    metode3 = tree.DecisionTreeClassifier(criterion="gini")
    metode3.fit(X_train, y_train)

    st.write("Pilih metode yang ingin anda gunakan :")
    met1 = st.checkbox("KNN")
    met2 = st.checkbox("Naive Bayes")
    met3 = st.checkbox("Decision Tree")

    if st.button("Pilih"):
        if met1:
            st.write("Metode yang Anda gunakan Adalah KNN")
            st.write("Hasil Akurasi Data Training Menggunakan KNN sebesar:", 100 * metode1.score(X_train, y_train))
            st.write("Hasil Akurasi Data Testing Menggunakan KNN sebesar:", 100 * metode1.score(X_test, y_test))
        elif met2:
            st.write("Metode yang Anda gunakan Adalah Naive Bayes")
            st.write("Hasil Akurasi Data Training Menggunakan Naive Bayes sebesar:", 100 * metode2.score(X_train, y_train))
            st.write("Hasil Akurasi Data Testing Menggunakan Naive Bayes sebesar:", 100 * metode2.score(X_test, y_test))
        elif met3:
            st.write("Metode yang Anda gunakan Adalah Decision Tree")
            st.write("Hasil Akurasi Data Training Menggunakan Decision Tree sebesar:", 100 * metode3.score(X_train, y_train))
            st.write("Hasil Akurasi Data Testing Menggunakan Decision Tree sebesar:", 100 * metode3.score(X_test, y_test))
        else:
            st.write("Anda Belum Memilih Metode")
