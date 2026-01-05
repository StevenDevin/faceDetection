# Face Recognition Attendance System

Panduan singkat untuk instalasi, setup environment, dan menjalankan program.

==================================================
1. PRASYARAT
==================================================
- Python 3.10.x 
- Webcam aktif

Cek versi Python:
python --version

Jika bukan 3.10, install dari:
https://www.python.org/downloads/release/python-3100/

==================================================
2. CLONE REPOSITORY
==================================================
git clone https://github.com/USERNAME/NAMA-REPO.git
cd NAMA-REPO

==================================================
3. BUAT VIRTUAL ENVIRONMENT
==================================================
python -m venv fr_env
fr_env\Scripts\activate

==================================================
4. INSTALL DEPENDENCY
==================================================
Upgrade pip:
pip install --upgrade pip

Install (ada di repo setelah download tinggal double click dan ikuti proses instalasi):
cmake-4.2.1-windows-x86_64
dlib-19.22.99-cp310-cp310-win_amd64
VC_redist.x64

Install semua library:
pip install -r requirements.txt

==================================================
5. SIAPKAN DATASET WAJAH
==================================================
Buat folder:
dataset_wajah

Masukkan foto wajah:
nama_01.jpg
nama_02.jpg

Catatan:
- 1 wajah per foto
- JPG / PNG
- Wajah harus jelas

==================================================
6. JALANKAN PROGRAM
==================================================
python main.py

==================================================
7. KONTROL PROGRAM
==================================================
Q  : Keluar
R  : Reset absensi

==================================================
8. TROUBLESHOOTING
==================================================
Python masih 3.11:
py -3.10 main.py




