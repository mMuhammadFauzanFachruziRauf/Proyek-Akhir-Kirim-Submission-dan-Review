## Domain Proyek

Pertumbuhan eksponensial konten digital seperti film dan acara TV di berbagai platform streaming (Netflix, Amazon Prime, Disney+, dan lainnya) memberikan kemudahan akses bagi pengguna. Namun, hal ini juga menimbulkan masalah yang disebut sebagai **_information overload_** atau **_paradox of choice_** — yaitu kesulitan dalam memilih konten yang sesuai dari ribuan pilihan yang tersedia.

Di sinilah **sistem rekomendasi** berperan penting. Sistem ini membantu pengguna menemukan konten yang relevan dan personal secara otomatis, sehingga:
- Meningkatkan pengalaman menonton pengguna.
- Mengurangi waktu pencarian konten.
- Meningkatkan engagement dan retensi pengguna pada platform streaming.

Proyek ini penting untuk diselesaikan karena memberikan solusi terhadap masalah _information overload_ tersebut, baik dari sisi pengguna yang mendapatkan kemudahan akses konten relevan, maupun dari sisi platform yang dapat meningkatkan interaksi penggunanya. Dengan memahami dan menerapkan teknik sistem rekomendasi, kita dapat membangun aplikasi cerdas yang memberikan nilai tambah signifikan.

### Referensi dan Studi Pendukung:

- Sarwar, B., Karypis, G., Konstan, J., & Riedl, J. (2001).  
  *Item-based collaborative filtering recommendation algorithms*.  
  *Proceedings of the 10th international conference on World Wide Web.*
  (https://dl.acm.org/doi/10.1145/371920.372071)

- Ricci, F., Rokach, L., & Shapira, B. (2011).  
  *Introduction to Recommender Systems Handbook*.  
  *Recommender Systems Handbook.*
  (Contoh link, bisa disesuaikan: https://link.springer.com/referenceworkentry/10.1007/978-0-387-85820-3_1)

- Gomez-Uribe, C. A., & Hunt, N. (2016). The netflix recommender system: Algorithms, business value, and innovation. *ACM Transactions on Management Information Systems (TMIS)*, 6(4), 1-19. (Ini contoh referensi dampak bisnis)

## Business Understanding

Bagian ini bertujuan untuk mengidentifikasi dan mengartikulasikan permasalahan yang ingin dipecahkan serta tujuan yang ingin dicapai melalui pengembangan sistem rekomendasi film.

### Problem Statements

1.  **Information Overload:**
    > Pengguna dihadapkan pada volume besar pilihan film yang tersedia pada platform digital, mengakibatkan kesulitan dalam menemukan film yang benar-benar sesuai dengan preferensi individu secara efisien. Fenomena ini dikenal sebagai *information overload* dan dapat menurunkan kualitas pengalaman pengguna.

2.  **Kurangnya Eksplorasi Film:**
    > Keterbatasan dalam eksplorasi film menyebabkan banyak karya sinematik, terutama yang berasal dari produksi independen atau rilisan lampau, menjadi kurang terekspos kepada audiens yang potensial, meskipun film tersebut mungkin relevan dengan minat mereka.

3.  **Rekomendasi yang Tidak Personal:**
    > Penyajian rekomendasi yang kurang personal atau tidak akurat dapat menyebabkan pengguna merasa frustrasi dan berpotensi mengurangi tingkat interaksi (*engagement*) mereka dengan layanan penyedia konten film.

### Goals

- Mengembangkan **dua model sistem rekomendasi film** — satu berbasis konten (*Content-Based Filtering*) dan satu berbasis kolaboratif (*Collaborative Filtering*) — untuk menghasilkan daftar rekomendasi film yang dipersonalisasi.
- Meningkatkan kemampuan pengguna dalam menemukan film baru yang sesuai dengan selera mereka, sehingga memperkaya pengalaman menonton dan meningkatkan unsur *serendipity* (penemuan tak terduga yang menyenangkan).
- Mengevaluasi dan membandingkan kinerja kedua pendekatan menggunakan **metrik evaluasi yang sesuai** pada dataset MovieLens.
- Memberikan pemahaman praktis mengenai implementasi dan evaluasi sistem rekomendasi sebagai salah
  satu aplikasi penting di bidang *machine learning*.

### Solution statements
Untuk mencapai tujuan tersebut, proyek ini akan mengimplementasikan dan mengevaluasi dua pendekatan utama dalam sistem rekomendasi:

1.  **Content-Based Filtering (Penyaringan Berbasis Konten):**
    * **Konsep Dasar:** Merekomendasikan film kepada pengguna berdasarkan kemiripan antara atribut atau 'konten' dari film-film tersebut (dalam proyek ini, fokus pada genre) dengan atribut film yang telah disukai oleh pengguna di masa lalu.
    * **Mekanisme Umum:**
        1.  Representasi Fitur Item: Setiap film direpresentasikan sebagai vektor fitur genre.
        2.  Pembuatan Profil Pengguna (implisit): Preferensi pengguna dimodelkan berdasarkan film yang mereka sukai.
        3.  Penghitungan Kemiripan: Cosine similarity digunakan untuk menghitung kemiripan antar film berdasarkan vektor genre.
        4.  Pemberian Rekomendasi: Film dengan skor kemiripan tertinggi terhadap film referensi akan direkomendasikan.
    * **Metrik Evaluasi (Kualitatif):** Relevansi genre pada film yang direkomendasikan.

2.  **Collaborative Filtering (Penyaringan Kolaboratif) menggunakan SVD:**
    * **Konsep Dasar:** Bekerja dengan mengidentifikasi pola dari data interaksi pengguna-item (rating). Rekomendasi dihasilkan berdasarkan preferensi dari sekelompok pengguna yang memiliki selera serupa atau berdasarkan kemiripan antar item. Algoritma SVD (Singular Value Decomposition), sebagai teknik faktorisasi matriks, digunakan untuk menemukan faktor laten pengguna dan item.
    * **Mekanisme Umum:**
        1.  Membangun matriks interaksi user-item (implisit dari data rating).
        2.  Menggunakan SVD untuk menguraikan matriks tersebut menjadi faktor laten pengguna dan item.
        3.  Memprediksi rating untuk pasangan user-item yang belum ada.
        4.  Merekomendasikan item dengan prediksi rating tertinggi.
    * **Metrik Evaluasi (Kuantitatif):** RMSE (Root Mean Squared Error) dan MAE (Mean Absolute Error) untuk mengukur akurasi prediksi rating.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah **MovieLens Latest Datasets (Small)** yang disediakan oleh GroupLens Research. Dataset ini berisi sekitar 100.000 rating dari sekitar 600 pengguna untuk sekitar 9.000 film.
-   Link Download: [https://grouplens.org/datasets/movielens/latest/](https://grouplens.org/datasets/movielens/latest/) (file `ml-latest-small.zip`).

Fokus utama adalah pada file `movies.csv` dan `ratings.csv`.

### Variabel-variabel pada dataset yang digunakan adalah sebagai berikut:

**Dari `movies.csv` (`movies_df`):**
* `movieId`: Identifier unik untuk setiap film dalam dataset.
    * *Tipe Data:* `int64` (Integer Numerik)
    * *Keterangan:* Digunakan sebagai kunci utama untuk menghubungkan data film dengan data rating.
* `title`: Judul lengkap dari film, seringkali menyertakan tahun rilis dalam tanda kurung.
    * *Tipe Data:* `object` (String)
* `genres`: Satu atau lebih genre yang diasosiasikan dengan film, dipisahkan oleh karakter pipa (`|`).
    * *Tipe Data:* `object` (String)
    * *Keterangan:* Fitur krusial untuk Content-Based Filtering.

**Dari `ratings.csv` (`ratings_df`):**
* `userId`: Identifier unik untuk setiap pengguna yang memberikan rating.
    * *Tipe Data:* `int64` (Integer Numerik)
* `movieId`: Identifier film yang diberi rating oleh pengguna.
    * *Tipe Data:* `int64` (Integer Numerik)
    * *Keterangan:* Merupakan foreign key yang menghubungkan ke `movieId` di `movies_df`.
* `rating`: Rating numerik yang diberikan oleh pengguna untuk film tertentu.
    * *Tipe Data:* `float64` (Float Numerik)
    * *Keterangan:* Skala rating 0.5 hingga 5.0. Variabel target utama untuk Collaborative Filtering.
* `timestamp`: Waktu ketika rating diberikan oleh pengguna (detik sejak epoch).
    * *Tipe Data:* `int64` (Integer Numerik)

### Exploratory Data Analysis (EDA)

Beberapa tahapan EDA dan visualisasi dilakukan untuk memahami data lebih dalam:

1.  **Informasi Umum Data:**
    * `movies_df` memiliki 9742 baris (film) dan 3 kolom. Tidak ada missing values.
    * `ratings_df` memiliki 100836 baris (rating) dan 4 kolom. Tidak ada missing values. Rata-rata rating adalah ~3.5.

2.  **Distribusi Rating Film:**
    * *Observasi:* Rating paling sering diberikan adalah 4.0, diikuti 3.0. Pengguna cenderung memberikan rating positif. Distribusi miring ke kiri.

3.  **Jumlah Film Unik dan Pengguna Unik:**
    * Terdapat 9742 film unik dan 610 pengguna unik.

4.  **Popularitas Film (Berdasarkan Jumlah Rating):**
    * *Observasi:* Film paling populer adalah "Forrest Gump (1994)" dengan 329 rating. Sebagian besar film hanya menerima sedikit rating (distribusi "long tail").

5.  **Aktivitas Pengguna (Berdasarkan Jumlah Film yang Dirating):**
    * *Observasi:* Rata-rata pengguna memberikan ~165 rating. Distribusi miring ke kanan, dengan mayoritas pengguna memberikan sedikit rating dan beberapa "power users" memberikan banyak rating. Pengguna paling aktif (ID 414) memberikan 2698 rating.

6.  **Analisis Genre Film:**
    * *Observasi:* Genre paling umum adalah Drama (4361 film) dan Comedy (3756 film). Terdapat 34 film dengan `(no genres listed)`.

7.  **Pemeriksaan Missing Values (Formal):**
    * Dikonfirmasi tidak ada missing values pada kolom-kolom utama `movies_df` dan `ratings_df` setelah pembersihan awal.

## Data Preparation

Beberapa teknik data preparation diterapkan untuk menyiapkan data sebelum modeling:

1.  **Menangani Film dengan Genre `(no genres listed)`:**
    * **Proses:** 34 film dengan genre `(no genres listed)` diidentifikasi. Film-film ini beserta rating terkaitnya dihapus dari `movies_df` dan `ratings_df`. Jumlah film menjadi 9708 dan jumlah rating menjadi 100789.
    * **Alasan:** Penting untuk kualitas fitur Content-Based Filtering dan konsistensi data.

2.  **Persiapan Data untuk Content-Based Filtering: Ekstraksi Fitur Genre:**
    * **Proses:** Kolom `genres` pada `movies_df` diubah menjadi representasi numerik menggunakan One-Hot Encoding (via `str.get_dummies()` dengan `sep='|'`). Setiap genre unik menjadi kolom biner (0 atau 1). Hasilnya adalah 19 kolom genre baru.
    * **Alasan:** Model machine learning memerlukan input numerik. Ini memungkinkan perhitungan kemiripan berdasarkan genre.

3.  **Persiapan Data untuk Collaborative Filtering: Pembagian Data (Train-Test Split):**
    * **Proses:** `ratings_df` (setelah pembersihan) dibagi menjadi set pelatihan (80%) dan set pengujian (20%) menggunakan `train_test_split` dari `sklearn.model_selection`. `train_ratings_df` memiliki 80631 rating, dan `test_ratings_df` memiliki 20158 rating.
    * **Alasan:** Untuk evaluasi model yang tidak bias, mencegah overfitting, dan dasar perbandingan model.

## Modeling

Dua pendekatan model machine learning digunakan: Content-Based Filtering dan Collaborative Filtering (SVD).

### A. Content-Based Filtering (CBF)

1.  **Perhitungan Kemiripan Film:**
    * **Tahapan:** Matriks fitur genre (9708 film x 19 genre unik) yang telah di-One-Hot Encode digunakan. Cosine Similarity dihitung antar semua pasangan film pada matriks genre ini, menghasilkan matriks kemiripan film-film (9708x9708).
    * **Parameter:** Tidak ada parameter tuning spesifik untuk `cosine_similarity` dalam konteks ini selain input matriks fiturnya.

2.  **Fungsi Rekomendasi CBF dan Hasilnya:**
    * **Tahapan:** Fungsi `get_movie_recommendations_cbf` dibuat untuk menerima judul film referensi dan mengembalikan Top-N film paling mirip berdasarkan skor dari matriks kemiripan.
    * **Contoh Hasil untuk 'Toy Story (1995)':**
      Genre asli `Toy Story (1995)`: `Adventure|Animation|Children|Comedy|Fantasy`  
      Berikut adalah 10 film yang direkomendasikan:

    | Judul Film                                       | Skor Kemiripan   | Genre                                             |
    | :----------------------------------------------- | :--------------- | :------------------------------------------------ |
    | Adventures of Rocky and Bullwinkle, The (2000)   | 1.0              | Adventure\|Animation\|Children\|Comedy\|Fantasy   |
    | Emperor's New Groove, The (2000)                 | 1.0              | Adventure\|Animation\|Children\|Comedy\|Fantasy   |
    | Monsters, Inc. (2001)                            | 1.0              | Adventure\|Animation\|Children\|Comedy\|Fantasy   |
    | Tale of Despereaux, The (2008)                   | 1.0              | Adventure\|Animation\|Children\|Comedy\|Fantasy   |
    | Wild, The (2006)                                 | 1.0              | Adventure\|Animation\|Children\|Comedy\|Fantasy   |
    | Moana (2016)                                     | 1.0              | Adventure\|Animation\|Children\|Comedy\|Fantasy   |
    | Turbo (2013)                                     | 1.0              | Adventure\|Animation\|Children\|Comedy\|Fantasy   |
    | Antz (1998)                                      | 1.0              | Adventure\|Animation\|Children\|Comedy\|Fantasy   |
    | Toy Story 2 (1999)                               | 1.0              | Adventure\|Animation\|Children\|Comedy\|Fantasy   |
    | The Good Dinosaur (2015)                         | 1.0              | Adventure\|Animation\|Children\|Comedy\|Fantasy   |

    *Observasi:* Film-film yang direkomendasikan untuk `Toy Story (1995)` semuanya memiliki skor kemiripan 1.0. Ini berarti mereka memiliki kombinasi genre yang **persis sama** (`Adventure|Animation|Children|Comedy|Fantasy`) dengan film referensi. Sistem ini efektif menemukan film-film dengan profil genre identik.

 * **Contoh Hasil untuk 'Matrix, The (1999)':**
    Genre asli `Matrix, The (1999)`: `Action|Sci-Fi|Thriller`  
    Berikut adalah 5 film yang direkomendasikan:

    | Judul Film                | Skor Kemiripan   | Genre                    |
    | :------------------------ | :--------------- | :----------------------- |
    | Total Recall (2012)       | 1.0              | Action\|Sci-Fi\|Thriller |
    | One, The (2001)           | 1.0              | Action\|Sci-Fi\|Thriller |
    | Déjà Vu (Deja Vu) (2006)  | 1.0              | Action\|Sci-Fi\|Thriller |
    | eXistenZ (1999)           | 1.0              | Action\|Sci-Fi\|Thriller |
    | Lockout (2012)            | 1.0              | Action\|Sci-Fi\|Thriller |
    
    *Observasi:* Rekomendasi untuk `Matrix, The (1999)` juga menunjukkan film-film dengan skor kemiripan 1.0, yang berarti semuanya memiliki genre `Action|Sci-Fi|Thriller`, sama persis dengan film referensi. Ini konsisten dengan pendekatan berbasis konten yang sangat bergantung pada kesamaan fitur yang didefinisikan.

3.  **Kelebihan dan Kekurangan CBF:**
    * **Kelebihan:** Independensi pengguna, transparansi, tidak ada cold start untuk item baru, mampu merekomendasikan item niche.
    * **Kekurangan:** Ketergantungan pada kualitas fitur, overspecialisasi (terutama jika fitur terbatas seperti hanya genre), cold start untuk pengguna baru, kesulitan menangkap nuansa.

### B. Collaborative Filtering (SVD)

1.  **Persiapan Data untuk Library `Surprise`:**
    * **Tahapan:** Data `train_ratings_df` dan `test_ratings_df` dimuat ke dalam format `Dataset` `Surprise` menggunakan `Reader` dengan skala rating 0.5-5.0. `trainset_surprise` (610 pengguna, 8956 item, 80631 rating) dan `testset_surprise` (20158 rating) dibuat.

2.  **Pelatihan Model SVD:**
    * **Tahapan:** Model `SVD` dari library `Surprise` diinisialisasi dan dilatih menggunakan `trainset_surprise`.
    * **Parameter Contoh:** `n_factors=100`, `n_epochs=25`, `lr_all=0.007`, `reg_all=0.05`, `random_state=42`. (Sebaiknya sebutkan parameter yang benar-benar Anda gunakan jika berbeda).

3.  **Fungsi Rekomendasi SVD dan Hasilnya:**
    * **Tahapan:** Fungsi `get_movie_recommendations_cf_svd` dibuat untuk menerima `user_id` dan mengembalikan Top-N film yang belum dirating pengguna tersebut, diurutkan berdasarkan prediksi rating tertinggi dari model SVD.
* **Contoh Hasil untuk User ID 1:**
    Berikut adalah 10 film yang direkomendasikan beserta prediksi ratingnya:

    | Judul Film                                        | Genre                                            | Prediksi Rating |
    | :------------------------------------------------ | :----------------------------------------------- | :-------------- |
    | Lawrence of Arabia (1962)                         | Adventure\|Drama\|War                            | 5.0             |
    | Godfather: Part II, The (1974)                    | Crime\|Drama                                     | 5.0             |
    | Touch of Evil (1958)                              | Crime\|Film-Noir\|Thriller                       | 5.0             |
    | Gladiator (1992)                                  | Action\|Drama                                    | 5.0             |
    | Three Billboards Outside Ebbing, Missouri (2017)  | Crime\|Drama                                     | 5.0             |
    | Yojimbo (1961)                                    | Action\|Adventure                                | 5.0             |
    | Great Escape, The (1963)                          | Action\|Adventure\|Drama\|War                    | 5.0             |
    | Glory (1989)                                      | Drama\|War                                       | 5.0             |
    | Grand Day Out with Wallace and Gromit, A (1989)   | Adventure\|Animation\|Children\|Comedy\|Sci-Fi   | 5.0             |
    | Godfather, The (1972)                             | Crime\|Drama                                     | 5.0             |
    
    *Observasi:* Rekomendasi untuk User ID `1` menunjukkan film-film dengan prediksi rating tertinggi, yaitu 5.0 untuk semua 10 film teratas. Ini mengindikasikan bahwa model SVD sangat percaya diri bahwa pengguna ini akan sangat menyukai film-film klasik dan yang mendapat pujian tinggi dari berbagai genre seperti Drama, War, Crime, dan Adventure.

* **Contoh Hasil untuk User ID 100:**
      Berikut adalah 5 film yang direkomendasikan beserta prediksi ratingnya:

    | Judul Film                                        | Genre                                            | Prediksi Rating |
    | :------------------------------------------------ | :----------------------------------------------- | :-------------- |
    | Yojimbo (1961)                                    | Action\|Adventure                                | 4.685115        |
    | Cookie's Fortune (1999)                           | Comedy\|Drama                                    | 4.678235        |
    | Cool Hand Luke (1967)                             | Drama                                            | 4.614846        |
    | Three Billboards Outside Ebbing, Missouri (2017)  | Crime\|Drama                                     | 4.612072        |
    | Grand Day Out with Wallace and Gromit, A (1989)   | Adventure\|Animation\|Children\|Comedy\|Sci-Fi   | 4.593617        |
    
    *Observasi:* Untuk User ID `100`, model SVD merekomendasikan film-film dengan prediksi rating yang juga sangat tinggi, berkisar antara 4.59 hingga 4.68. Rekomendasi ini juga mencakup berbagai genre, menunjukkan kemampuan SVD untuk menangkap preferensi yang mungkin lebih beragam atau nuansa yang tidak hanya berdasarkan genre eksplisit seperti pada Content-Based Filtering.

4.  **Kelebihan dan Kekurangan CF (SVD):**
    * **Kelebihan:** Tidak memerlukan analisis konten, mampu menemukan rekomendasi yang mengejutkan (serendipity), mempelajari pola preferensi kompleks, relatif baik menangani sparsitas (untuk SVD).
    * **Kekurangan:** Masalah cold start (pengguna baru & item baru), potensi bias popularitas, ketergantungan pada jumlah data interaksi, kurang transparan.

## Evaluation

Evaluasi model dilakukan untuk mengukur performa sistem rekomendasi yang dibangun.

### Evaluasi Model Content-Based Filtering (CBF)

### Evaluasi Model Content-Based Filtering (CBF)

Evaluasi untuk model Content-Based Filtering (CBF) yang dibangun dalam proyek ini mencakup aspek **kualitatif** dan **kuantitatif**.

#### a. Evaluasi Kualitatif

* **Pendekatan:** Observasi relevansi hasil rekomendasi dengan membandingkan genre film yang direkomendasikan dengan film input.
* **Hasil dan Interpretasi (Kualitatif):** Model CBF, yang hanya menggunakan fitur genre, secara konsisten merekomendasikan film dengan kombinasi genre yang persis sama (skor kemiripan cosine 1.0) dengan film referensi. Ini menunjukkan model bekerja sesuai desainnya dalam mengidentifikasi item berdasarkan kesamaan maksimal pada fitur genre yang didefinisikan. Sebagai contoh, untuk film 'Toy Story (1995)' (genre: `Adventure|Animation|Children|Comedy|Fantasy`), semua 10 film teratas yang direkomendasikan memiliki set genre yang identik. Hal serupa terjadi untuk 'Matrix, The (1999)' (genre: `Action|Sci-Fi|Thriller`).

#### b. Evaluasi Kuantitatif: Precision@K

Meskipun CBF murni berbasis kemiripan item tidak secara langsung memprediksi rating, kemampuannya untuk merekomendasikan item yang relevan kepada pengguna dapat diukur menggunakan metrik seperti Precision@K.

* **Pendekatan untuk Precision@K:**
    1.  Menggunakan `test_ratings_df` sebagai dasar "ground truth".
    2.  Untuk setiap pengguna dalam `test_ratings_df`, diambil satu film yang mereka rating tinggi (rating >= 4.0) sebagai *query item* (film referensi).
    3.  Model CBF memberikan rekomendasi Top-K (misalnya K=5 atau K=10) film berdasarkan kemiripan genre dengan *query item* tersebut.
    4.  Memeriksa berapa banyak dari K rekomendasi tersebut yang juga merupakan film *lain* yang dirating tinggi (rating >= 4.0) oleh pengguna yang sama dalam `test_ratings_df`.
    5.  Precision@K dihitung sebagai: (Jumlah item relevan yang direkomendasikan dalam Top-K) / K.
    6.  Nilai Precision@K rata-rata di seluruh pengguna uji dilaporkan.

* **Metrik yang Digunakan:**
    * **Precision@K**: Mengukur proporsi item yang direkomendasikan dalam K item teratas yang benar-benar relevan bagi pengguna.
        $$ \text{Precision@K} = \frac{\text{Jumlah item relevan dan direkomendasikan dalam Top-K}}{K} $$
       

* **Hasil dan Interpretasi (Kuantitatif):**
    Berdasarkan perhitungan pada notebook:
    * Nilai **Precision@10** yang diperoleh adalah: **0.0037**
    * Nilai **Precision@5** yang diperoleh adalah: **0.0032**

    Interpretasi hasil ini:
    * Nilai Precision@K yang diperoleh untuk model CBF yang hanya berbasis kemiripan genre ini terbilang sangat rendah.
    * Sebagai contoh, untuk Precision@10, dari 10 film yang direkomendasikan berdasarkan kemiripan genre dengan satu film yang disukai pengguna, rata-rata hanya sekitar 0.037 film (kurang dari 1 film dari 100 pengguna uji) yang ternyata juga dinilai relevan (dirating >= 4.0) oleh pengguna tersebut di test set.
    * Ini mengindikasikan bahwa kemiripan genre saja, meskipun menghasilkan film-film yang secara konten serupa, memiliki keterbatasan signifikan dalam merekomendasikan item yang akan disukai pengguna berdasarkan riwayat rating mereka yang lebih luas. Model ini tidak mempertimbangkan preferensi rating pengguna lain atau nuansa preferensi individu di luar kesamaan genre.

* **Keterbatasan Evaluasi CBF (Berdasarkan Genre Saja):**
    * Evaluasi kualitatif menunjukkan model bekerja sesuai desainnya untuk menemukan film dengan genre serupa.
    * Evaluasi kuantitatif (Precision@K) yang rendah menyoroti bahwa kesamaan genre saja tidak cukup untuk menjamin relevansi tinggi menurut preferensi rating pengguna yang lebih luas. Model CBF ini tidak secara langsung mengukur kepuasan pengguna atau memprediksi rating, dan performanya dalam menyarankan film yang relevan (berdasarkan rating tinggi) sangat terbatas ketika hanya mengandalkan genre.

### Evaluasi Model Collaborative Filtering (SVD)

### Metrik yang Digunakan:

1. **Root Mean Squared Error (RMSE)**  
   RMSE mengukur akar kuadrat dari rata-rata selisih kuadrat antara rating aktual dan prediksi. Metrik ini memberikan penalti lebih besar terhadap error yang besar.

   **Rumus:**

   $$
   \text{RMSE} = \sqrt{ \frac{1}{N} \sum_{(u,i) \in \text{TestSet}} \left( r_{ui} - \hat{r}_{ui} \right)^2 }
   $$

2. **Mean Absolute Error (MAE)**  
   MAE mengukur rata-rata dari selisih absolut antara rating aktual dan prediksi. Metrik ini lebih toleran terhadap outlier dibanding RMSE.

   **Rumus:**

   $$
   \text{MAE} = \frac{1}{N} \sum_{(u,i) \in \text{TestSet}} \left| r_{ui} - \hat{r}_{ui} \right|
   $$

**Proses Perhitungan:**
Model SVD terlatih digunakan untuk membuat prediksi rating pada `testset_surprise`. Fungsi `accuracy.rmse()` dan `accuracy.mae()` dari `Surprise` menghitung error berdasarkan perbandingan rating aktual dan prediksi.

**Hasil dan Interpretasi:**
* Nilai **RMSE** yang diperoleh adalah: **0.8628**
* Nilai **MAE** yang diperoleh adalah: **0.6618**

Interpretasi hasil ini:
* Secara rata-rata, prediksi rating dari model SVD memiliki kesalahan absolut sebesar 0.6618 poin dari rating aktual (pada skala 0.5-5.0).
* RMSE 0.8628, sedikit lebih tinggi dari MAE, mengindikasikan adanya beberapa prediksi dengan error yang lebih besar. Namun, kedua nilai ini menunjukkan performa yang cukup baik untuk baseline awal.
* Nilai ini bisa menjadi acuan untuk perbaikan model di masa depan (misalnya, dengan hyperparameter tuning).

**---Ini adalah bagian akhir laporan---**
