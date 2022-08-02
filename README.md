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
movie_id	title	overview	genres	keywords	cast	crew
0	19995	Avatar	In the 22nd century, a paraplegic Marine is di...	[{"id": 28, "name": "Action"}, {"id": 12, "nam...	[{"id": 1463, "name": "culture clash"}, {"id":...	[{"cast_id": 242, "character": "Jake Sully", "...	[{"credit_id": "52fe48009251416c750aca23", "de...
1	285	Pirates of the Caribbean: At World's End	Captain Barbossa, long believed to be dead, ha...	[{"id": 12, "name": "Adventure"}, {"id": 14, "...	[{"id": 270, "name": "ocean"}, {"id": 726, "na...	[{"cast_id": 4, "character": "Captain Jack Spa...	[{"credit_id": "52fe4232c3a36847f800b579", "de...
2	206647	Spectre	A cryptic message from Bond’s past sends him o...	[{"id": 28, "name": "Action"}, {"id": 12, "nam...	[{"id": 470, "name": "spy"}, {"id": 818, "name...	[{"cast_id": 1, "character": "James Bond", "cr...	[{"credit_id": "54805967c3a36829b5002c41", "de...
3	49026	The Dark Knight Rises	Following the death of District Attorney Harve...	[{"id": 28, "name": "Action"}, {"id": 80, "nam...	[{"id": 849, "name": "dc comics"}, {"id": 853,...	[{"cast_id": 2, "character": "Bruce Wayne / Ba...	[{"credit_id": "52fe4781c3a36847f81398c3", "de...
4	49529	John Carter	John Carter is a war-weary, former military ca...	[{"id": 28, "name": "Action"}, {"id": 12, "nam...	[{"id": 818, "name": "based on novel"}, {"id":...	[{"cast_id": 5, "character": "John Carter", "c...	[{"credit_id": "52fe479ac3a36847f813eaa3", "de...




berikut adalah contoh gambar aksara sunda huruf ba yang akan kita ubah kedalam bentuk tensor supaya data dapat dikenali oleh model. Dengan kita mengubahnya kedalam tensor, kita juga dapat mengekstraksi fitur yang ada pada gambar.

![gambar fitur](https://user-images.githubusercontent.com/106476815/181236794-5a64370f-263c-4633-a23f-a90f0d87b3db.png)

((354, 354),
array([[1., 1., 1., ..., 1., 1., 1.],

[1., 1., 1., ..., 1., 1., 1.],
        
[1., 1., 1., ..., 1., 1., 1.],
        
...,
        
[1., 1., 1., ..., 1., 1., 1.],
        
[1., 1., 1., ..., 1., 1., 1.],
        
[1., 1., 1., ..., 1., 1., 1.]]))
        
Pada gambar awal, gambar berukuran 354x354 dan hitam putih. Fitur ekstrak ini menghasilkan sebuah angka dimana setiap angka merepresentasikan sebuah warna.


## Data Preparation

Data train / data latih akan digunakan untuk melatih model, sedangkan data test akan digunakan sebagai validation untuk model. Pada data train akan dilakukan normalisasi dengan membagi (rescale) semuanya dengan 1/255. Lalu gambar juga akan dilakukan augmentasi. Tetapi untuk data validation hanya dilakukan normalisasi saja. Lalu tiap data train dan validation akan diubah warnanya biar sama. Dengan ImageDataGenerator kita dapat memproduksi berbagai varisasi data tanpa memakan atau menggunakan 'space' penyimpanan kita, sehingga model dapat lebih belajar banyak variasi data, seperti foto pada gambar diperbesar, dibalik secara horizontal maupun vertikal, tetapi karena ini tulisan aksara kita tidak menggunakan flip karena artinya akan berbeda nanti.


## Modeling
Pada awal layer, kita menggunakan conv2D untuk membentuk lapisan konvolusi karena data yang kita masukkan berupa tensor 2 dimensi kita menggunakan aktivasi relu atau Rectified Linear Unit karena keuntungannya yaitu mempercepat proses konvergensi yang dilakukan dengan stochastic gradient descent jika dibandingkan dengan sigmoid / tanh dan padding agar semua memiliki ukuran yang sama, tidak lupa juga untuk lapisan pertama argumen input_shape, (112,112,1) karena gambar input kita memiliki ukuran 100x100 dan warna hanya hitam putih jadi kita menulisnya 100x100x1. Lalu kita menulis lapisan berikutnya seperti LeakyReLU, bisa dilakukan pemanggilan atau aktivasi layer lain, hasil ini diperoleh dari trial and error. Lalu MaxPooling2D untuk operasi pooling. Lalu GlobalMaxPool2D lalu Dense, pada akhir layer digunakan activation softmax karena klasifikasi yang kita lakukan lebih dari 2 dan output 18 karena kita mempunyai 18 kelas data. Untuk model digunakan optimasi Adam, karena label kita akan berupa one-hot-encoded, kita menggunakan categorical_crossentropy, lalu dengan metrik yang melakukan judge pada model melihat dari akurasi yang dihasilkan.


## Evaluation

![acc](https://user-images.githubusercontent.com/106476815/181580647-3aa65748-9514-4764-9b2f-71c43ae9e660.png)
![loss](https://user-images.githubusercontent.com/106476815/181580636-831aaab5-6d09-4519-a4c0-422c95f0db15.png)

loss: 0.0179 - accuracy: 0.9944 - val_loss: 0.0032 - val_accuracy: 0.9988
loss adalah nilai yang didapat dari hasil model melakukan training menggunakan data training sedangkan val_loss adalah nilai yang didapat dari hasil model menggunakan data validasi. Keduanya memiliki arti yang sama yaitu menilai seberapa buruk model memprediksi suatu hal, semakin baik model maka nilai loss dan val_loss akan bernilai makin kecil atau bahkan mendekati 0.  
accuracy merupakan nilai dari hasil model yang dilatih menggunakan data latih, sedangkan val_accuracy merupakan nilai dari hasil model memprediksi sampel yang tidak ikut terlatih atau kita bisa sebut seberapa besar akurasi model jika digunakan pada kasus nyata.

![metric](https://user-images.githubusercontent.com/106476815/181580642-909dc65a-b98c-4b11-afcb-7112a78aed4b.png)

Pada klasifikasi gambar kita menggunakan nilai accuracy sebagai metrik, kita mendapatkan hasil yang tinggi. Jika kita lihat juga dari nilai accuracy dan val_accuracy yang hampir serupa bahkan mendekati 100%, dan juga nilai loss serta val_loss yang sama-sama mendekati 0, maka kita dapat katakan model ini sudah sangat baik dalam melakukan klasifikasi tulisan aksara Sunda.
Sehingga diharapkan dengan model ini, pelajar dapat mengetahui apakah tulisan aksara Sunda yang ia tulis sudah tepat atau belum.

## References
[1]Kirmemis, Oznur, and Aysenur Birturk. "A content-based user model generation and optimization approach for movie recommendation." Workshop on ITWP. 2008.
[2]J. Salter and N. Antonopoulos, "CinemaScreen recommender agent: combining collaborative and content-based filtering," in IEEE Intelligent Systems, vol. 21, no. 1, pp. 35-41, Jan.-Feb. 2006, doi: 10.1109/MIS.2006.4.
