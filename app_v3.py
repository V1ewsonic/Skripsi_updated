#Import Library
import os
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import swifter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import seaborn as sns
import plotly.graph_objects as px
import matplotlib.pyplot as plt
from PIL import Image
import io
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')


#-----------------------Function-----------------------#

# Fungsi untuk mengupload data ke website
def data_load(datas):

    ext = os.path.splitext(datas.name)[1].lower()

    if ext == ".csv":
        data = pd.read_csv(datas)
    elif ext == ".json":
        data = pd.read_json(datas)
    elif ext in [".xls", ".xlsx"]:
        data = pd.read_excel(datas)
    else:
        raise ValueError("Format file tidak didukung. Gunakan CSV, JSON, atau Excel.")

    return data


def change_name(df,kata):
    '''
    Mengganti nama kolom yang dimasukkan oleh user menjadi 'komentar'
    agar mudah diproses oleh sintaks yang telah dibaut
    '''
#     nama_kolom = input('Masukkan nama kolom anda : ')
    kolom_baru = 'komentar'
    df.rename(columns={nama_kolom: kolom_baru}, inplace=True)
    # df = df.head()
    return df

def preprocessed(df):
    '''
    Melakuakan pre-processing data untuk FILTERING dan CASEFOLDING
    '''
#     %%time
    # Import Library yang dibutuhkan
    import datetime as dt
    import re
    import string
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import swifter
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

    # Filtering
    #@jit(nopython=True)
    def filtering(text):
        text = re.sub('[0-9]+', '', str(text)) #removing numberic value
        text = re.sub(r'#', '', text) #removing '#' symbol
        text = re.sub(r'[\n]+', '', text) # remove new line
        text = re.sub(r"^\s+|\s+$", "", text) #remove leading and trailing spaces in a word using OR sign to delete both
        text = re.sub(r" +", " ", text) #remove multiple space betwen words
        text = re.sub('https? :\/\/\S+', '', text) #removing hyperlink / URL
        text = re.sub(r"\b[a-zAZ]\b", "", text) #removing single char
        text = re.sub('\s+',' ',text) #removing multiple whitespace
        text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"") #remove tab, new line, and back slice
        text = re.sub(r'[^\w\s]', '', text) #remove puntuation& emoji (remove all besides \w > word dan \s > space)
        # text = re.sub(r'(.)1+', r'1', text) # remove repeating character
        text = re.sub("[^a-zA-Z]",' ',text)

        return text

    #@jit(nopython=True)
    def casefoldingText(text): # Converting all the characters in a text into lower case
        text = text.lower()
        return text

    #Data Cleaning Process
#     %%time
    df['filtering'] = df['komentar'].swifter.apply(filtering)
#     %%time
    df['casefolding'] = df['filtering'].swifter.apply(casefoldingText)

    # Hasil perubahan Filtering dan Case Folding
    print('Komentar Awal :\n', df['komentar'].iloc[5],'\n')
    print('Sesudah Filtering :\n', df['filtering'].iloc[5],'\n')
    print('Sesudah Case Folding :\n', df['casefolding'].iloc[5])
    return df

def training_lda(df):

  # Menjadikan LIST
  text_list = []
  for i in df['casefolding']:
    if isinstance(i, str):  # Memastikan nilai adalah objek string
        text_list.append(i.split())

  # TRAINING SEKALIGUS MEMBUAT FILE CSV
  # Memisahkan Topic 1 dan 2

  import pandas as pd
  from sklearn.feature_extraction.text import CountVectorizer
  from sklearn.decomposition import LatentDirichletAllocation

  # Membuat list untuk menyimpan hasil training
  doc_list = []
  topic1_list = []
  topic2_list = []

  # Loop untuk menyimpan setiap baris data menjadi list
  for doc_index, doc in enumerate(text_list):
      print(f"Processing Document {doc_index + 1}: {doc}")

      # Vectorization
      vectorizer = CountVectorizer()
      X = vectorizer.fit_transform([' '.join(doc)])

      # LDA model fitting
      num_topics = 2
      lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
      lda.fit(X)

      # Mendapatkan top keyword untuk setiap hasil LDA
      feature_names = vectorizer.get_feature_names_out()
      topics = []
      for topic_idx, topic in enumerate(lda.components_):
          top_keywords = [feature_names[i] for i in topic.argsort()[:-5-1:-1]]
          topics.append(', '.join(top_keywords))

      # Menyimpan hasil trainng
      doc_list.append(doc)
      topic1_list.append(topics[0])
      topic2_list.append(topics[1])

  # Buat DataFrame dari List
  df_results = pd.DataFrame({'Teks': doc_list, 'Topic 1': topic1_list, 'Topic 2': topic2_list})

  # Menyimpan hasil training LDA
  df_results.to_csv('output3.csv', index=False)

  # Menggabungkan Topic 1 dan 2 
  df_result = pd.read_csv('output3.csv')
  df_result['merged_column'] = df_result['Topic 1'].str.cat(df_result['Topic 2'], sep=', ')

  # Menyimpan hasil merged_column menjadi CSV file
  df_result.to_csv('output3_2.csv', index=False)

  # Fungsi untuk mengecek apakah ada kata yang tergolong 'negatif' DataFrame
  def check_negatif(row):
    topic_words = row['merged_column'].split(", ")  # Split the words in the string
    for word in topic_words:
        if word.strip().lower() in negatif_['NEG'].str.lower().str.strip().values:
            return True
    return False


# Fungsi pengecekkan kata objek
  def check_object(row2):
    object_words = row2['merged_column'].split(", ")
    for Word in object_words:
      if Word.strip().lower() in objek_['OBJ'].str.lower().str.strip().values:
        return True
      return False

  # Membaca data kumpulan kata 'Negatif'
  negatif_ = pd.read_csv('negatif_word.csv')

  # Membaca data kumpulan kata 'Objek'
  objek_ = pd.read_csv('object_word.csv')

  # Membaca data hasil Topic Modelling
  output4 = pd.read_csv('output3_2.csv')

  # Mengaplikasikan fungsi yang telah dibuat sebelumnya pada kolom 'Neg' in 'output4'
  output4['Neg'] = output4.apply(check_negatif, axis=1)

  # Mengaplikasikan fungsi yang telah dibuat sebelumnya pada kolom 'Objek' in 'output4'
  output4['Objek'] = output4.apply(check_object, axis=1)

  # Banyaknya jumlah Negatif
  output4['sentimen'] = output4['Neg'].apply(lambda x: 'Negatif' if x == True else 'Positif')
  sent_negatif = output4['Neg'].sum()
  # Banyaknya jumlah Positif
  sent_positif= output4['sentimen'].eq('Positif').sum()
  # Menampilkan hasil sentimen Topic Model
  st.write('Jumlah Sentimen Negatif : ',{sent_negatif})
  st.write('Jumlah Sentimen Positif : ',{sent_positif})
  
  output4 = output4[['Teks','sentimen']]
  return output4

# Fungsi pengecekkan apakah format yang diupload sesuai kriteria
def is_valid_data(filename):
    return filename.lower().endswith(('.csv', '.json', '.xls', '.xlsx'))

# Fungsi untuk mengupload data ke website
def data_load_clust(datas):
    # Mendapatkan ekstensi file
    if hasattr(datas, 'name'):
        ext = os.path.splitext(datas.name)[1].lower()
    else:
        ext = ".csv" # Default jika nama tidak terbaca

    # Logika pembacaan berdasarkan ekstensi
    if ext == ".csv":
        df = pd.read_csv(datas)
    elif ext == ".json":
        df = pd.read_json(datas)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(datas)
    else:
        raise ValueError("Format file tidak didukung.")

    # Logika pemotongan kolom
    banyak_kolom = len(df.axes[1])
    data = df.iloc[:, 1:banyak_kolom]
    return data

# Fungsi untuk menyimpan chart menjadi Gambar
def fig2img(fig):
  import io
  from PIL import Image
  buf = io.BytesIO()
  fig.savefig(buf)
  buf.seek(0)
  img = Image.open(buf)
  return img


# Fungsi untuk membuat cluster data Likert
def elbow_plot(data):
  # Import library yang hanya dibutuhkan
  import matplotlib.pyplot as plt
  import seaborn as sns
  # Optimasi K-Means dengan metode elbow untuk menentukan jumlah klaster yang tepat
  from sklearn.cluster import KMeans
  from sklearn import metrics
  from sklearn.metrics import pairwise_distances
  from sklearn.cluster import KMeans
  from sklearn.metrics import silhouette_samples, silhouette_score
  # me-non aktifkan peringatan pada python
  import warnings
  warnings.filterwarnings('ignore')

  # Proses Plotting untuk menentukan jumlah cluster terbaik
  wcss = []
  for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

  plt.plot(range(1, 11), wcss)
  plt.title('Elbow Method')
  plt.xlabel('Cluster Number')
  plt.ylabel('WCSS')
  plt.show()

  # Getting the current figure and save it in the variable.
  fig = plt.gcf()

  # Mengkonversi gambar dengan fungsi fig2img yang telah dibuat.
  img = fig2img(fig)
  
# Menyimpan gambar dengan bantuan fungsi save().
  img.save('elbow.png')  
  return img

def evaluasi_cluster(data):
  # Pemrosesan nilai Silhouette
  import numpy as np
  from sklearn.cluster import KMeans
  from sklearn import metrics

  range_n_clusters = range(2, 11)
  best_silhouette_score = -1
  best_n_clusters = -1

  # Proses Plotting untuk menentukan jumlah cluster terbaik
  for i in range_n_clusters:
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    cluster_labels = kmeans.fit_predict(data)
    sil_avg = metrics.silhouette_score(data, cluster_labels, metric='euclidean')
    
    print("Untuk jumlah cluster =", i, "rata-rata nilai silhouette nya adalah :", sil_avg)
    
    if sil_avg > best_silhouette_score:
        best_silhouette_score = sil_avg
        best_n_clusters = i

  a = print("Jumlah cluster terbaik berdasarkan silhouette score:", best_n_clusters)
  return a

def visualisasi_cluster(data):
  # Pemrosesan nilai Silhouette
  import numpy as np
  from sklearn.cluster import KMeans
  from sklearn import metrics

  range_n_clusters = range(2, 11)
  best_silhouette_score = -1
  best_n_clusters = -1

  # Proses Plotting untuk menentukan jumlah cluster terbaik
  for i in range_n_clusters:
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    cluster_labels = kmeans.fit_predict(data)
    sil_avg = metrics.silhouette_score(data, cluster_labels, metric='euclidean')
    
    print("Untuk jumlah cluster =", i, "rata-rata nilai silhouette nya adalah :", sil_avg)
    
    if sil_avg > best_silhouette_score:
        best_silhouette_score = sil_avg
        best_n_clusters = i

  # Training Cluster
  kmeans_best = KMeans(n_clusters=best_n_clusters, init='k-means++', random_state=42)
  y_kmeans = kmeans_best.fit_predict(data)

  # Menghitung persebaran Cluster yang ada
  data["Clusters"] = y_kmeans

  #Plotting countplot dari clusters
  pal = ["#40E0D0","#B9C0C9", "#DE3163","#F3AB60","#73c771","#b87bc9"]
  pl = sns.countplot(x=data["Clusters"], palette= pal)
  pl.set_title("Distribusi dari Clusters")
  plt.show()

    # Getting the current figure and save it in the variable.
  fig2 = plt.gcf()

  # Mengkonversi gambar dengan fungsi fig2img yang telah dibuat.
  img2 = fig2img(fig2)
  
# Menyimpan gambar dengan bantuan fungsi save().
  img2.save('hasil_cluster.png')  
  return img2

  # Menghitung persebaran cluster pada data
  # clust = data["Clusters"].value_counts()
  # print("Banyaknya Cluster : \n",clust)
  # return data
#   pass


def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')



# Sidebar
menu_sidebar = st.sidebar.selectbox('Modeling',('Clustering - Data Ordinal','Topic Modeling - Komentar',
                                                'Hasil Analisis Sentimen'))
if menu_sidebar == 'Topic Modeling - Komentar':
#    Title Halaman Topic Modeling
     st.title(':green[Topic Modeling - Komentar]')

#    Pengantar Halaman
     st.header('Panduan')
     st.markdown('''1. Gunakan format file CSV/Excel/JSON \
                 \n2. Pastikan kolom yang akan anda proses bertipe data _string_ atau _object_ \
                 \n3. Masukkan nama kolom yang akan di proses \
                 \n4. Klik tombol 'Latih Data' untuk mendapatkan hasil sentimen
                    ''')
     
#    Upload file ke Website
     uploaded_file = st.file_uploader('Upload File Anda')

     if uploaded_file is not None:
       #  Kondisi file harus CSV
          file_name = uploaded_file.name

          if is_valid_data(file_name):
               df = data_load(uploaded_file)
               st.dataframe(df)

     #         Data Pre-processing        
               nama_kolom = st.text_input('Masukkan Nama Kolom yang berisikan komentar: ')

               if not nama_kolom :
                    st.info("Silahkan Masukkan Nama Kolom Yang Berisikan Komentar")
                    st.stop()
               else:
                    ubah_kolom = change_name(df,kata=nama_kolom)
                    olah_data = preprocessed(df)
                    df_olah = df[['komentar','filtering','casefolding']]
                    st.write('Data setelah di bersihkan')
                    st.dataframe(df_olah)

#                   Train Data
                    if st.button('Latih Data'):
                         st.subheader(':green[Sedang Melatih Model...]  :sunglasses:')
                         LDA_train = training_lda(df)
                         output_training = LDA_train
                         output_training.to_csv('topic_model.csv')
                         st.dataframe(output_training)
                    
          else:
               st.error('Mohon upload file berekstensi .csv')
     else: 
          st.info("Silahkan upload file CSV Anda")
          st.stop()
if menu_sidebar == 'Clustering - Data Ordinal':
    #    Title Halaman Topic Modeling
    st.title(':blue[Clustering - Data Ordinal]')

#    Pengantar Halaman
    st.header('Panduan')
    st.markdown('''1. Gunakan format file CSV/Excel/JSON \
                 \n2. Pastikan kolom yang akan anda proses bertipe data _integer_ (angka) \
                 \n3. Pastikan file csv hanya berisikan kolom untuk hasil skala Likert \
                 \n4. Klik tombol 'Latih Data' untuk mendapatkan hasil _cluster_
                    ''')
#    Upload file ke Website
    uploaded_file = st.file_uploader('Upload File Data', type=['csv', 'json', 'xlsx', 'xls'])

    if uploaded_file is not None:
       #  Kondisi file harus CSV
        file_name = uploaded_file.name

        if is_valid_data(file_name):
            df = data_load_clust(uploaded_file)
            st.dataframe(df)

#           Train Data
            if st.button('Latih Data'):
                st.subheader(':green[Sedang Melatih Model...]  :sunglasses:')
                wcss = []
                for i in range(1, 11):
                  kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
                  kmeans.fit(df)
                  wcss.append(kmeans.inertia_)

                plt.figure(figsize=(6, 4))
                plt.plot(range(1, 11), wcss)
                plt.title('Elbow Method')
                plt.xlabel('Cluster Number')
                plt.ylabel('WCSS')
                plt.show()
                plt.savefig('elbow.png')
                plt.close()

               
                image = Image.open('elbow.png')
                st.image(image, caption='Cluster terbaik berdasarkan Metode Elbow')
                
                # Pemrosesan nilai Silhouette
                range_n_clusters = range(2, 11)
                best_silhouette_score = -1
                best_n_clusters = -1

                # Proses Plotting untuk menentukan jumlah cluster terbaik
                for i in range_n_clusters:
                  kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
                  cluster_labels = kmeans.fit_predict(df)
                  sil_avg = metrics.silhouette_score(df, cluster_labels, metric='euclidean')
                  
                  st.write("Untuk jumlah cluster =", i, "rata-rata nilai silhouette nya adalah :", sil_avg)
                  
                  if sil_avg > best_silhouette_score:
                      best_silhouette_score = sil_avg
                      best_n_clusters = i

                st.write("Jumlah cluster terbaik berdasarkan silhouette score:", best_n_clusters)
                
#               Training Cluster
                kmeans_best = KMeans(n_clusters=best_n_clusters, init='k-means++', random_state=42)
                y_kmeans = kmeans_best.fit_predict(df)

                # Menghitung persebaran Cluster yang ada
                df["Clusters"] = y_kmeans

                #Plotting countplot dari clusters
                plt.figure(figsize=(6, 4))
                pal = ["#40E0D0","#B9C0C9", "#DE3163","#F3AB60","#73c771","#b87bc9"]
                pl = sns.countplot(x=df["Clusters"], palette= pal)
                pl.set_title("Distribusi dari Clusters")
                plt.show()
                plt.savefig('cluster.png')
                plt.close()

                image2 = Image.open('cluster.png')
                st.image(image2, caption='Cluster terbaik berdasarkan Metode Elbow')

                clust = df["Clusters"].value_counts()
                st.write("Banyaknya Cluster : \n",clust)
                df.to_csv('data_cluster_likert.csv')                  
        else:
            st.error('Mohon upload file berekstensi .csv')
    else: 
        st.info("Silahkan upload file Data Anda")
        st.stop()

if menu_sidebar == 'Hasil Analisis Sentimen':
  st.title(':violet[Hasil Analisis Sentimen]')

   #  Pengantar Halaman
  st.header('Panduan')
  st.markdown('''1. Lakukan Training Topic Model dan Clustering \
                 \n2. Klik tombol 'Lakukan Analisis Sentimen' untuk mendapatkan hasil pelatihan
                    ''')
  st.warning("Pastikan Anda sudah melakukan Tahap 1 Clustering - Data Ordinal dan Tahap 2 Topic Modeling - Komentar")

  # Button latih data
  if st.button('Lakukan Analisis Sentimen'):
    st.subheader(':green[Sedang Melatih Model...]  :sunglasses:')
   # Import library yang dibutuhkan 
    import pandas as pd
    from sklearn import preprocessing
    

  # Import Data
    klaster = pd.read_csv('data_cluster_likert.csv')
    topic = pd.read_csv('topic_model.csv')
    

# label encoding pada hasil sentimen tahap 2
    label_encoder = preprocessing.LabelEncoder()
    topic['sentimen']= label_encoder.fit_transform(topic['sentimen'])
    topic['sentimen'].unique()

# Menyatukan 2 dataframe menjadi satu
    newdf = pd.concat([klaster['Clusters'], topic['sentimen']], axis=1)
    X = topic['sentimen']
    y = klaster['Clusters']


    # FINAL , apabila silhouette score = 1 tidak akan diambil sebagai cluster
    # terbaik

    best_silhouette_score = -1  # Initialize with a lower value than possible Silhouette scores
    best_n_clusters = -1
    range_n_clusters = range(2,11)

    for i in range_n_clusters:
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        cluster_labels = kmeans.fit_predict(newdf)
        sil_avg = metrics.silhouette_score(newdf, cluster_labels, metric='euclidean')

        print("Untuk jumlah cluster =", i, "rata-rata nilai silhouette nya adalah :", sil_avg)

        if sil_avg > best_silhouette_score:
            if sil_avg == 1:
                best_silhouette_score = prev_sil_avg  # Use the previous Silhouette score value
            else:
                best_silhouette_score = sil_avg
                best_n_clusters = i

        prev_sil_avg = sil_avg  # Store the current Silhouette score for the next iteration

    st.write("Jumlah cluster terbaik berdasarkan silhouette score:", best_n_clusters)

    # Training Tahap 3
    kmeans_best_ = KMeans(n_clusters=best_n_clusters, init='k-means++', random_state=42)
    y_kmeans_ = kmeans_best_.fit_predict(newdf)

    # Visualisasi hasil Training
    # Menghitung persebaran Cluster yang ada
    newdf["Klaster"] = y_kmeans_

    #Plotting countplot dari clusters
    pal = ["#40E0D0","#B9C0C9", "#DE3163","#F3AB60"]
    pl = sns.countplot(x=newdf["Klaster"], palette= pal)
    pl.set_title("Distribusi dari Clusters")
    plt.show()
    plt.savefig('cluster3.png')
    plt.close()

    #Menampilkan gambar Tahap 3
    image3 = Image.open('cluster3.png')
    st.image(image3, caption='Hasil clustering Tahap 3')


    # Define a mapping for the replacement
    replacement_map = {0: 'Netral', 1: 'Positif', 2: 'Negatif'}

#   Replace values in the DataFrame
    newdf['Sentimen'] = newdf['Klaster'].replace(replacement_map)
    final_df = pd.concat([topic['Teks'],newdf['Sentimen']], axis=1)

    clust3 = final_df['Sentimen'].value_counts()
    st.write("Hasil Cluster : \n",clust3)


    # Button Download
    csv = convert_df(final_df)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='hasil_analisis_sentimen.csv',
        mime='text/csv',
    )

    




