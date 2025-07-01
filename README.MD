# Langkah - Langkah Untuk Menghubungkan DJI Tello dengan Program ini
> **Catatan:** Penggunaan Joystick hanya bisa digunakan di Linux

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
