# DJI Tello Control and Autonomous Navigation

## Kontrol dan Navigasi Autonomous DJI Tello: Integrasi Gamepad, Deteksi Objek YOLOv8, dan Autonomous berbasis Pendeteksian warna merah

### Langkah - Langkah Untuk Menghubungkan DJI Tello dengan Program ini
> **Catatan:** Penggunaan Joystick hanya bisa digunakan di Linux.

## Langkah Instalasi

1. **Clone repository**
    ```bash
    git clone https://github.com/GeryTrstno/protel-code.git
    ```

2. **Masuk ke direktori repository**
    ```bash
    cd protel-code
    ```

3. **Buat virtual environment**
    ```bash
    python -m venv venv
    ```

4. **Aktifkan virtual environment**
    - **Windows:**
      ```bash
      venv\Scripts\activate
      ```
    - **Linux/MacOS:**
      ```bash
      source venv/bin/activate
      ```

5. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

6. **Masuk ke folder `Final`**
    ```bash
    cd Final
    ```

> **Catatan:** Pastikan komputer Anda sudah terkoneksi dengan WiFi DJI Tello sebelum menjalankan aplikasi.

7. **Jalankan aplikasi**
    ```bash
    python main.py
    ```

### Fitur Utama

- **Integrasi Pergerakan DJI Tello Menggunakan Controller Gamepad**: Mengontrol DJI Tello dengan menggunakan joystick untuk navigasi yang lebih intuitif.
  
- **Integrasi Machine Learning untuk Deteksi Objek**: Menggunakan YOLOv8 dan Mediapipe untuk mendeteksi objek secara real-time.

- **Pergerakan Otonom Berbasis Pendeteksian Warna Merah**: DJI Tello dapat bergerak secara otomatis berdasarkan deteksi warna merah di lingkungan sekitarnya.

### Catatan Tambahan

- Pastikan semua perangkat keras dan perangkat lunak yang diperlukan telah terinstal dan dikonfigurasi dengan benar sebelum menjalankan aplikasi.
- Untuk informasi lebih lanjut tentang penggunaan YOLOv8, Mediapipe, dan SDK Tello silakan merujuk ke dokumentasi resmi masing-masing.
