# --------------------------
#   1. Temel imaj
# --------------------------
FROM python:3.10-slim

# --------------------------
#   2. Sistem bağımlılıkları
# --------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# --------------------------
#   3. Çalışma klasörü
# --------------------------
WORKDIR /app

# --------------------------
#   4. Python bağımlılıkları
# --------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------
#   5. Proje dosyalarını kopyala
# --------------------------
COPY . .

# --------------------------
#   6. .env dosyasını kopyala
# --------------------------
COPY .env .

# --------------------------
#   7. Streamlit portu
# --------------------------
EXPOSE 8501

# --------------------------
#   8. Transformers cache klasörleri
# --------------------------
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache

# --------------------------
#   9. Streamlit başlat
# --------------------------
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
