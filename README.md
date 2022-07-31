# Laporan_Submission_2_Hansen Jonathan

## Domain Proyek
Personalisasi telah menjadi pendekatan yang kuat untuk membangun informasi yang lebih tepat dan mudah digunakan sistem pencarian dan rekomendasi. Kualitas dari personalisasi sangat bergantung pada keakuratan model pengguna yang dibuat oleh sistem dan itu sangat penting untuk memasukkan informasi konten dari domain kerja untuk memperkaya model ini[1].Agen pemberi rekomendasi film memperluas dan menyempurnakan hasil penyaringan kolaboratif menurut elemen konten yang disaring - yaitu, aktor, sutradara, dan genre. Pendekatan ini mendukung rekomendasi untuk judul yang baru dirilis, yang sebelumnya belum diberi rating. Mengarahkan pengguna ke konten yang relevan semakin penting dalam masyarakat saat ini dengan massa informasi yang terus berkembang. Untuk tujuan ini, sistem rekomendasi telah menjadi komponen penting dari sistem e-commerce dan domain aplikasi yang menarik untuk teknologi agen cerdas[2]. Netflix merupakan salah satu industri yang menyajikan rekomendasi film pada aplikasinya dan menurut data dari yang mereka miliki banyak sekali orang-orang yang menonton film-film yang ada di Netflix karena hasil dari rekomendasi film yang diasjikan. Hal ini tentu sangat menguntungkan juga untuk beberapa sektor bisnis maupun industri lainnya dengan menerapkan sistem rekomendasi. Dengan sistem rekomendasi ini, diharapkan juga pengalaman kita menonton film yang kita sukai karena aktor, sutradara ataupun genre film tersebut semakin memuaskan dan sesuai dengan selera kita.

## Business Understanding
1. Problem Statement:
   Bagaimana supaya orang dapat menulis dan belajar sendiri aksara Sunda dengan tepat?
2. Goals:
   Menciptakan machine learning model yang mampu melakukan klasifikasi penulisan aksara Sunda dengan tepat. Dengan klasifikasi tulisan yang pelajar tulis, ia dapat memahami apakah yang ia tulis sudah benar atau belum. Sehingga dengan hal ini, pelajar dapat belajar sendiri penulisan aksara Sunda dengan tepat


## Data Understanding
Data yang digunakan berasal dari https://github.com/ridhomujizat/AksaraSundaCNN/.
Data Loading: jika kita melihat pada data lokal, terdapat 18 kelas masing-masing pada data train dan test.
18 kelas tersebut terdiri dari:

![ra](https://user-images.githubusercontent.com/106476815/181581155-23cc3a81-8d32-43e6-add9-7f78a76ce4c5.png)

ra

![sa](https://user-images.githubusercontent.com/106476815/181581162-8420022e-eeeb-4649-ba7b-48f2cc4bcd82.png)

sa

![ta](https://user-images.githubusercontent.com/106476815/181581167-558560d1-2327-43fa-89a0-3f96303b97b5.png)

ta

![wa](https://user-images.githubusercontent.com/106476815/181581169-3a509c79-ca07-4478-8916-e02c8e8e06eb.png)

wa

![ya](https://user-images.githubusercontent.com/106476815/181581171-8a4c77bc-9fe2-49de-a980-9d61788b5b4d.png)

ya

![ba](https://user-images.githubusercontent.com/106476815/181581175-65c0637d-c0e2-4fea-b689-3c581fc363da.png)

ba

![ca](https://user-images.githubusercontent.com/106476815/181581176-1d3ed2f3-f31d-4ffa-81b3-83fc26e48531.png)

ca

![da](https://user-images.githubusercontent.com/106476815/181581182-ef9c2115-eef3-412b-a696-ec99866486f6.png)

da

![ga](https://user-images.githubusercontent.com/106476815/181581185-06faa381-8e86-466c-849f-51258f39c3c5.png)

ga

![ha](https://user-images.githubusercontent.com/106476815/181581186-ba185280-8837-45b7-a2ec-c3154a70cbae.png)

ha

![ja](https://user-images.githubusercontent.com/106476815/181581189-d7a19e9b-8de8-4025-9413-d1929b0e9664.png)

ja

![ka](https://user-images.githubusercontent.com/106476815/181581191-d4df8c00-5037-4abc-a694-0215029e2b0e.png)

ka

![la](https://user-images.githubusercontent.com/106476815/181581193-7a88a41f-d2ae-48b2-9794-55ab0260db02.png)

la

![ma](https://user-images.githubusercontent.com/106476815/181581197-76fa3ddc-f653-456d-b4b9-d712b7101722.png)

ma

![na](https://user-images.githubusercontent.com/106476815/181581198-c31017b4-cae3-48f9-a0f4-4bd458269ca3.png)

na

![nga](https://user-images.githubusercontent.com/106476815/181581202-b3daa361-4097-4328-bdbe-81535f45cf4d.png)

nga

![nya](https://user-images.githubusercontent.com/106476815/181581207-877db8a5-436d-4a4e-85bb-5be7f8da0663.png)

nya

![pa](https://user-images.githubusercontent.com/106476815/181581209-55969aa5-2f3d-4222-8b06-afa946a98bbf.png)

pa

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
