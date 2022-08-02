# Laporan_Submission_2_Hansen Jonathan

## Domain Proyek
Personalisasi telah menjadi pendekatan yang kuat untuk membangun informasi yang lebih tepat dan mudah digunakan sistem pencarian dan rekomendasi. Kualitas dari personalisasi sangat bergantung pada keakuratan model pengguna yang dibuat oleh sistem dan itu sangat penting untuk memasukkan informasi konten dari domain kerja untuk memperkaya model ini[1].Agen pemberi rekomendasi film memperluas dan menyempurnakan hasil penyaringan kolaboratif menurut elemen konten yang disaring - yaitu, aktor, sutradara, dan genre. Pendekatan ini mendukung rekomendasi untuk judul yang baru dirilis, yang sebelumnya belum diberi rating. Mengarahkan pengguna ke konten yang relevan semakin penting dalam masyarakat saat ini dengan massa informasi yang terus berkembang. Untuk tujuan ini, sistem rekomendasi telah menjadi komponen penting dari sistem e-commerce dan domain aplikasi yang menarik untuk teknologi agen cerdas[2]. Netflix merupakan salah satu industri yang menyajikan rekomendasi film pada aplikasinya dan menurut data dari yang mereka miliki banyak sekali orang-orang yang menonton film-film yang ada di Netflix karena hasil dari rekomendasi film yang disajikan. Hal ini tentu sangat menguntungkan juga untuk beberapa sektor bisnis maupun industri lainnya dengan menerapkan sistem rekomendasi. Dengan sistem rekomendasi ini, diharapkan juga pengalaman kita menonton film yang kita sukai karena aktor, sutradara ataupun genre film tersebut semakin memuaskan dan sesuai dengan selera kita.

## Business Understanding
1. *Problem Statement*:
   Bagaimana supaya orang dapat menonton film-film yang mungkin saja mereka tidak tahu sebelumnya, tetapi film tersebut sesuai dengan selera mereka?
2. *Goals*:
   Menciptakan machine learning model yang mampu merekomendasikan film yang sesuai dengan selera mereka. Sehingga film-film yang mungkin mereka tidak tahu sebelumnya tetapi film tersebut sesuai dengan selera mereka, dapat mereka ketahui.


## Data Understanding
Data yang digunakan berasal dari https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata.
Data Loading: jika kita melihat pada data tersebut terdapat 2 file yaitu movies dan credit dalam bentuk csv. 2 file tersebut kita *merge* tetapi, kita hanya membutuhkan beberapa kolom saja untuk model ini, yaitu *'movie_id','title','overview','genres','keywords','cast','crew'* lalu kita akan lihat 5 data teratas menggunakan *head(5)*

![data 5](https://user-images.githubusercontent.com/106476815/182332376-f1ce6827-381c-4abe-99a0-09643cab2030.jpg)

Sebelum kita olah lebih lanjut, kita perlu memastikan apakah data yang kita punya tidak punya nilai kosong atau N/A.

movie_id    0
title       0
overview    3
genres      0
keywords    0
cast        0
crew        0
dtype: int64

Jika kita lihat terdapat nilai kosong, pada kasus ini akan kita drop. Lalu kita cek kembali

movie_id    0
title       0
overview    0
genres      0
keywords    0
cast        0
crew        0
dtype: int64

Sekarang data kita sudah siap untuk diolah lebih lanjut.


## Data Preparation
Jika kita perhatikan pada kolom *genres, keywords, cast,* dan juga *crew*, penulisan teks masih berantakan untuk itu kita harus merapikan terlebih dahulu. 
Berikut adalah hasil setelah kolom-kolom tersebut dirapihkan teksnya.

![setelah konversi teks](https://user-images.githubusercontent.com/106476815/182333610-478930a9-4dcd-48aa-bf14-739e02066e7f.jpg)

Lalu kolom-kolom tersebut kita jadikan satu sehingga menjadi seperti gambar dibawah ini.

![digabungkan dalam tags](https://user-images.githubusercontent.com/106476815/182333603-eb6d3a4a-ef58-4b1b-b9d8-7f8ba8e1f31a.jpg)


## Modeling
Sampai sini, kita baru mengubah/mengkonversi teks pada kolom *tags* ke dalam sebuah token matriks. Pada kali ini kita menggunakan *CountVectorizer()*, setelah itu kita melakukan *fit_transform*, *fit_transform* merupakan kombinasi metode *fit()* dan *transform()* pada kumpulan data yang sama untuk transformasi dataset. Sekarang kita menggunakan teknik *cosine similarity* dari library sklearn. Kita coba plot hasilnya akan seperti berikut,

array([[1.        , 0.08964215, 0.06071767, ..., 0.02519763, 0.0277885 ,
        0.        ],
       [0.08964215, 1.        , 0.06350006, ..., 0.02635231, 0.        ,
        0.        ],
       [0.06071767, 0.06350006, 1.        , ..., 0.02677398, 0.        ,
        0.        ],
       ...,
       [0.02519763, 0.02635231, 0.02677398, ..., 1.        , 0.07352146,
        0.04774099],
       [0.0277885 , 0.        , 0.        , ..., 0.07352146, 1.        ,
        0.05264981],
       [0.        , 0.        , 0.        , ..., 0.04774099, 0.05264981,
        1.        ]])
        
Dengan cosine similarity, kita berhasil mengidentifikasi kesamaan antara satu film dengan film lainnya. Nilai-nilai tersebut sangat beragam karena tags yang kita gunakan untuk kemiripan sangat beragam dan banyak. Lalu kita akan buat modelnya yang akan kita panggil untuk memberi rekomendasi dari film yang kita berikan, disini kita akan memberi 10 rekomendasi film.

![model rekomen](https://user-images.githubusercontent.com/106476815/182336742-bc7a2f20-9436-4ad8-a105-b99ca8ada09f.jpg)
        

## Evaluation
Selanjutnya kita disini mencoba menemukan rekomendasi film yang mirip dengan *Pirates of the Caribbean: At World's End* dengan menjalankan kode berikut.

recommend("Pirates of the Caribbean: At World's End")

Ketika dijalankan maka akan menghasilkan sebagai berikut.

![output](https://user-images.githubusercontent.com/106476815/182336747-f7d8eec5-fd32-42f2-a864-a56757e55b13.jpg)

Dari hasil tersebut, *goals* yang kita inginkan sudah tercapai. Dapat dilihat juga, karena *Pirates of the Caribbean* sendiri mempunyai beberapa *sequel* film, sehingga tentu sistem/model akan merekomendasikan juga sekuel dari film tersebut.


## References
[1]Kirmemis, Oznur, and Aysenur Birturk. "A content-based user model generation and optimization approach for movie recommendation." Workshop on ITWP. 2008.
[2]J. Salter and N. Antonopoulos, "CinemaScreen recommender agent: combining collaborative and content-based filtering," in IEEE Intelligent Systems, vol. 21, no. 1, pp. 35-41, Jan.-Feb. 2006, doi: 10.1109/MIS.2006.4.
