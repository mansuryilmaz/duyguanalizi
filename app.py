import streamlit as st
import pandas as pd
import tweepy
import re
import string
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import time
import emoji
import os
from dotenv import load_dotenv

# --- 1. ENV Dosyasını Yükle ---
load_dotenv()

# --- 2. BEARER TOKEN ---
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

# --- 3. Türkçe Duygu Modeli ---
MODEL = "savasy/bert-base-turkish-sentiment-cased"
try:
    @st.cache_resource
    def load_model():
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL)
        return tokenizer, model
    
    tokenizer, model = load_model()
    labels = ["Olumsuz", "Nötr", "Olumlu"]
    MODEL_LOADED = True
except Exception as e:
    st.error(f"⚠️ Model yüklenirken hata: {e}")
    MODEL_LOADED = False

# --- 4. Twitter API Bağlantısı ---
try:
    if BEARER_TOKEN:
        client = tweepy.Client(bearer_token=BEARER_TOKEN)
        CLIENT_CONNECTED = True
    else:
        st.error("⚠️ Bearer Token tanımlanmadı veya .env dosyasından okunamadı.")
        CLIENT_CONNECTED = False
except Exception as e:
    st.error(f"❌ API bağlantı hatası: {e}")
    CLIENT_CONNECTED = False

# --- 5. Temizleme Fonksiyonu (Emoji dahil) ---
def clean_tweet(text):
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[0-9]+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Emoji temizleme
    text = emoji.replace_emoji(text, replace="")
    
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- 6. Duygu Tahmini ---
def predict_sentiment(text):
    if not MODEL_LOADED or not text:
        return "Nötr"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    return labels[torch.argmax(probs).item()]

# --- 7. Tweet Çekme ---
@st.cache_data(ttl=600)
def fetch_tweets(keyword, count):
    if not CLIENT_CONNECTED:
        st.error("API bağlantısı yok.")
        return None

    tweets_data = []
    max_count = min(count, 100)
    for attempt in range(3):
        try:
            response = client.search_recent_tweets(
                query=keyword + " lang:tr -is:retweet",
                max_results=max_count,
                tweet_fields=["created_at", "public_metrics"]
            )
            if response.data:
                for tweet in response.data:
                    tweets_data.append({
                        "Konu": keyword,
                        "tweet": tweet.text,
                        "RT_Sayısı": tweet.public_metrics.get("retweet_count", 0),
                        "Beğeni_Sayısı": tweet.public_metrics.get("like_count", 0)
                    })
                return pd.DataFrame(tweets_data)
            else:
                return pd.DataFrame()
        except tweepy.errors.TooManyRequests:
            time.sleep(60)
        except Exception as e:
            st.error(f"Hata: {e}")
            return None
    st.error("Veri çekilemedi.")
    return None

# --- 8. Streamlit Arayüzü ---
st.set_page_config(page_title="Sosyal Medya Duygu Analizi", layout="wide")
st.title("📊 Türkçe Duygu Analizi (Dosya Kaydetmeli)")

keyword = st.text_input("Analiz edilecek konu veya hashtag:", "İNÖNÜ ÜNİVERSİTESİ")
tweet_limit = st.slider("Kaç tweet çekilsin?", 10, 100, 20)

# 1️⃣ Tweetleri çek ve kaydet
if st.button("📥 Tweetleri Çek ve Kaydet"):
    if not keyword:
        st.warning("Bir konu giriniz.")
        st.stop()
    if not CLIENT_CONNECTED:
        st.warning("API'ye bağlanılamıyor.")
        st.stop()

    with st.spinner(f"‘{keyword}’ ile ilgili tweetler çekiliyor..."):
        df = fetch_tweets(keyword, tweet_limit)
    if df is None or df.empty:
        st.error("Tweet bulunamadı.")
        st.stop()

    df.to_csv("tweets.csv", index=False, encoding="utf-8-sig")
    st.success(f"{len(df)} tweet çekildi ve kaydedildi ✅")
    st.dataframe(df, use_container_width=True, height=500)

    csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode('utf-8')
    st.download_button(
        label="⬇️ Tweetleri CSV Olarak İndir",
        data=csv_bytes,
        file_name="tweets.csv",
        mime="text/csv"
    )

# 2️⃣ Kaydedilen CSV’yi modele ver ve analiz et
st.markdown("---")
st.subheader("🤖 Kaydedilen Dosyayı Model ile Analiz Et")
if st.button("▶️ Analiz Et (tweets.csv dosyasından)"):

    try:
        df = pd.read_csv("tweets.csv")
    except FileNotFoundError:
        st.error("Önce tweetleri çekip kaydetmelisin.")
        st.stop()

    df["clean_tweet"] = df["tweet"].apply(clean_tweet)
    df["Duygu"] = df["clean_tweet"].apply(predict_sentiment)

    df.to_csv("results.csv", index=False, encoding="utf-8-sig")
    st.success("Analiz tamamlandı ✅")
    st.dataframe(df, use_container_width=True, height=500)

    csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode('utf-8')
    st.download_button(
        label="⬇️ Analiz Sonuçlarını CSV Olarak İndir",
        data=csv_bytes,
        file_name="results.csv",
        mime="text/csv"
    )

    # Grafiksel Gösterim (Pasta Grafiği)
    st.subheader("📊 Duygu Dağılımı")
    counts = df["Duygu"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=["red", "gray", "green"]
    )
    ax.axis("equal")  # Daire şeklinde göstermek için
    st.pyplot(fig)
