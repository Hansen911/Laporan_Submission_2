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
Data Loading: jika kita melihat pada data tersebut terdapat 2 file yaitu movies dan credit dalam bentuk csv. 2 file tersebut kita *merge*.

![1](https://user-images.githubusercontent.com/106476815/182641398-d172a4c8-e1f3-49e2-b7a5-6a08a02491f7.jpg)

![2](https://user-images.githubusercontent.com/106476815/182641374-7a3e3981-d4e0-473d-8e32-f8b17c4b2fc3.jpg)

![3](https://user-images.githubusercontent.com/106476815/182641388-a816b50c-c6ec-4133-9097-0341b22081d7.jpg)

- budget = uang yang dikeluarkan untuk melakukan produksi film.
- genres = kategori film
- homepage = website sebagai sumber data
- id = nomor identitas  
- keywords = kata kunci film 
- original_language = bahasa asli yang digunakan film
- original_title = judul asli film
- overview = narasi singkat film
- popularity = ketenaran film
- production_companies = perusahaan pembuat film 
- production_countries = tempat(negara) film dibuat
- release_date = tanggal film terbit
- revenue = penghasilan film
- runtime = durasi film
- spoken_languages = bahasa yang digunakan dalam film
- status = keadaan film seperti sudah terbit atau belum
- tagline = tulisan singkat mengenai ide film
- title = judul film
- vote_average = rata-rata suara dari hasil voting film
- vote_count = banyaknya suara dari hasil voting film
- movie_id = nomor identitas film
- cast = pemain film
- crew = sutradara film

Tetapi kita hanya membutuhkan beberapa kolom saja untuk model ini, yaitu *'movie_id','title','overview','genres','keywords','cast','crew'* lalu kita akan lihat 5 data teratas menggunakan *head(5)*

![data 5](https://user-images.githubusercontent.com/106476815/182332376-f1ce6827-381c-4abe-99a0-09643cab2030.jpg)

Sebelum kita olah lebih lanjut, kita perlu memastikan apakah data yang kita punya tidak punya nilai kosong atau N/A.

![isna1](https://user-images.githubusercontent.com/106476815/182641391-fca7f091-86fb-4357-9480-be147b3a149e.jpg)

Jika kita lihat terdapat nilai kosong, pada kasus ini akan kita drop. Lalu kita cek kembali

![isna2](https://user-images.githubusercontent.com/106476815/182641396-d48b6da8-66f3-4481-8a50-dcf12cd15aa2.jpg)

Sekarang data kita sudah siap untuk diolah lebih lanjut.


## Data Preparation
Kali ini kita akan mencoba untuk menelusuri beberapa kolom yang ada.

![uuu](https://user-images.githubusercontent.com/106476815/182652484-3ae47733-a666-4484-8306-705785ed53d6.jpg)

![uuuuu](https://user-images.githubusercontent.com/106476815/182652468-45b1aebf-e6a2-45dc-a994-5f3ae05bdc2b.jpg)

![uuuuuuu](https://user-images.githubusercontent.com/106476815/182652478-69de0d04-bc21-40ea-b51a-0c00dffc0e3e.jpg)

Kita juga dapat mencoba untuk melihat film-film apa saja yang dimainkan oleh seorang Johnny Depp.

![uuuuuuuuu](https://user-images.githubusercontent.com/106476815/182652480-9d2e4f25-edea-4cb8-b294-cf1c4df67ba7.jpg)


Jika kita perhatikan pada kolom *genres, keywords, cast,* dan juga *crew*, penulisan teks masih berantakan untuk itu kita harus merapikan terlebih dahulu. 
Berikut adalah hasil setelah kolom-kolom tersebut dirapihkan teksnya.

![setelah konversi teks](https://user-images.githubusercontent.com/106476815/182333610-478930a9-4dcd-48aa-bf14-739e02066e7f.jpg)

Lalu kolom-kolom tersebut kita jadikan satu sehingga menjadi seperti gambar dibawah ini.

![digabungkan dalam tags](https://user-images.githubusercontent.com/106476815/182333603-eb6d3a4a-ef58-4b1b-b9d8-7f8ba8e1f31a.jpg)


## Modeling
Sampai sini, kita baru mengubah/mengkonversi teks pada kolom *tags* ke dalam sebuah token matriks. Pada kali ini kita menggunakan *CountVectorizer()*, setelah itu kita melakukan *fit_transform*, *fit_transform* merupakan kombinasi metode *fit()* dan *transform()* pada kumpulan data yang sama untuk transformasi dataset. Sekarang kita menggunakan teknik *cosine similarity* dari library sklearn. Berikut adalah hasil yang didapatkan ketika kita mengaplikasikan teknik tersebut.

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

Selanjutnya kita disini mencoba menemukan rekomendasi film yang mirip dengan *Pirates of the Caribbean: At World's End* dengan menjalankan kode berikut.

recommend("Pirates of the Caribbean: At World's End")

Ketika dijalankan maka akan menghasilkan sebagai berikut.

![output](https://user-images.githubusercontent.com/106476815/182336747-f7d8eec5-fd32-42f2-a864-a56757e55b13.jpg)

Dari hasil tersebut, *goals* yang kita inginkan sudah tercapai. Dapat dilihat juga, karena *Pirates of the Caribbean* sendiri mempunyai beberapa *sequel* film, sehingga tentu sistem/model akan merekomendasikan juga sekuel dari film tersebut.

## Evaluation



## References
[1]Kirmemis, Oznur, and Aysenur Birturk. "A content-based user model generation and optimization approach for movie recommendation." Workshop on ITWP. 2008.

[2]J. Salter and N. Antonopoulos, "CinemaScreen recommender agent: combining collaborative and content-based filtering," in IEEE Intelligent Systems, vol. 21, no. 1, pp. 35-41, Jan.-Feb. 2006, doi: 10.1109/MIS.2006.4.
