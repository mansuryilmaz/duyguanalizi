# 🇹🇷 Türkçe Sosyal Medya Duygu Analizi

📊 Bu proje, Twitter’dan çekilen Türkçe tweetleri analiz ederek Olumsuz / Nötr / Olumlu duygu sınıflarına ayırır. Sonuçları CSV olarak kaydedebilir ve grafiksel olarak görüntüleyebilirsiniz.

## Özellikler

- Twitter API ile tweet çekme  
- Emoji ve özel karakterleri temizleme  
- Hugging Face BERT tabanlı Türkçe duygu analizi  
- Streamlit arayüzü ile kullanıcı dostu kullanım  
- Sonuçları CSV olarak kaydetme ve indirme  
- Duygu dağılımını pasta grafiği ile görselleştirme

## Gereksinimler

Python 3.11+ ile çalışır. Gerekli Python paketleri:

streamlit, pandas, tweepy, torch, transformers, matplotlib, emoji, python-dotenv

## Kullanım

1. `.env` dosyasını proje klasörüne ekle. İçeriği şu şekilde olmalı:

    BEARER_TOKEN=twitter_bearer_token


2. Streamlit uygulamasını çalıştır:

   streamlit run app.py

3. Arayüzde analiz etmek istediğin konu veya hashtag’i gir.  
4. Tweet sayısını seç ve “📥 Tweetleri Çek ve Kaydet” butonuna bas.  
5. Ardından “▶️ Analiz Et” butonuna tıklayarak duygu analizini başlat.  
6. Sonuçları tablo olarak görüntüleyebilir, CSV olarak indirebilirsin.

## Proje Yapısı

duyguanalizi/  
├── app.py
├── .env
├── .gitignore    
└── README.md      


## Geliştirici
Mansur Yılmaz
Kerem Söyler

