import streamlit as st
import pandas as pd
import tweepy
import re
import string
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import emoji
import os
import redis
from io import BytesIO
from googleapiclient.discovery import build
import requests

# =====================================================
# 0. ENV (Streamlit Secrets / .env)
# =====================================================
def get_env(key, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

BEARER_TOKEN = get_env("BEARER_TOKEN")
YT_API_KEY = get_env("YT_API_KEY")
NEWS_API_KEY = get_env("NEWS_API_KEY")

REDIS_HOST = get_env("REDIS_HOST")
REDIS_PORT = int(get_env("REDIS_PORT", 6379))
REDIS_PASSWORD = get_env("REDIS_PASSWORD")
REDIS_SSL = get_env("REDIS_SSL", "false") == "true"

# =====================================================
# 1. REDIS
# =====================================================
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=True,
    ssl=REDIS_SSL,
    socket_connect_timeout=5
)

try:
    redis_client.ping()
except Exception as e:
    st.error(f"❌ Redis bağlantı hatası: {e}")
    st.stop()

# =====================================================
# 2. MODEL
# =====================================================
MODEL = "codealchemist01/turkish-sentiment-analysis"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    return tokenizer, model

tokenizer, model = load_model()
labels = ["Olumsuz", "Nötr", "Olumlu"]

# =====================================================
# 3. TWITTER CLIENT
# =====================================================
twitter_client = tweepy.Client(bearer_token=BEARER_TOKEN) if BEARER_TOKEN else None

# =====================================================
# 4. YOUTUBE CLIENT
# =====================================================
youtube = build("youtube", "v3", developerKey=YT_API_KEY)

# =====================================================
# 5. TEXT CLEAN
# =====================================================
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[0-9]+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = emoji.replace_emoji(text, replace="")
    return re.sub(r"\s+", " ", text).strip()

# =====================================================
# 6. SENTIMENT
# =====================================================
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    return labels[torch.argmax(probs).item()]

# =====================================================
# 7. TWITTER FETCH
# =====================================================
@st.cache_data(ttl=600)
def fetch_tweets(keyword, count):
    tweets = []
    if not twitter_client:
        return tweets
    try:
        res = twitter_client.search_recent_tweets(
            query=f"{keyword} lang:tr -is:retweet",
            max_results=max(10, min(count, 100))  # 10-100 arası olmalı
        )
        if res and res.data:
            tweets = [t.text for t in res.data]
    except Exception as e:
        st.warning(f"Twitter verisi alınamadı: {e}")
    return tweets

# =====================================================
# 8. YOUTUBE FETCH
# =====================================================
def get_video_ids(keyword, max_results=5):
    res = youtube.search().list(
        q=keyword,
        part="id",
        type="video",
        maxResults=max_results
    ).execute()
    return [item["id"]["videoId"] for item in res["items"]]

@st.cache_data(ttl=600)
def fetch_youtube_comments(keyword, limit):
    comments = []
    for vid in get_video_ids(keyword):
        res = youtube.commentThreads().list(
            part="snippet",
            videoId=vid,
            maxResults=100,
            textFormat="plainText"
        ).execute()
        for item in res["items"]:
            comments.append(item["snippet"]["topLevelComment"]["snippet"]["textDisplay"])
            if len(comments) >= limit:
                return comments
    return comments

# =====================================================
# 9. NEWS FETCH (düzeltildi)
# =====================================================
@st.cache_data(ttl=600)
def fetch_news(keyword, limit):
    url = f"https://newsapi.org/v2/everything?q={keyword}&language=tr&apiKey={NEWS_API_KEY}"
    try:
        res = requests.get(url).json()
        articles = res.get("articles", [])[:limit]
        result = []
        for a in articles:
            title = a.get("title","Başlık yok")
            url_ = a.get("url","#")
            result.append({"Başlık": title, "URL": url_})
        return result
    except Exception as e:
        st.warning(f"Haber verisi alınamadı: {e}")
        return []

# =====================================================
# Redis yardımcı fonksiyonlar
# =====================================================
def save_to_redis(key, data):
    redis_client.delete(key)
    for d in data:
        redis_client.lpush(key, str(d))
    redis_client.expire(key, 600)

def load_news_from_redis(key):
    data = redis_client.lrange(key, 0, -1)
    result = []
    for item in data:
        try:
            result.append(eval(item))
        except:
            continue
    return pd.DataFrame(result)

# =====================================================
# 10. THEME + CSS (tasarım aynen)
# =====================================================
mode = st.sidebar.radio("🎨 Tema", ["Light", "Dark"])
BG = "#0f172a" if mode == "Dark" else "#f8fafc"
CARD = "#020617" if mode == "Dark" else "#ffffff"
TEXT = "#e5e7eb" if mode == "Dark" else "#020617"
BORDER = "#1e293b" if mode == "Dark" else "#e5e7eb"

st.markdown(f"""
<style>
body, .block-container {{background:{BG}; color:{TEXT};}}
.card {{background:{CARD}; padding:20px; border-radius:18px;
border:1px solid {BORDER}; margin-bottom:20px;}}

.stButton>button {{
width:100%; border-radius:14px;
background:linear-gradient(90deg,#0284c7,#22d3ee);
font-weight:700; color:black;
}}

.kpi-box {{
    padding:12px;
    border-radius:14px;
    text-align:center;
    font-weight:700;
    color:white;
    margin-bottom:20px;
}}
.kpi-total {{ background:#2563eb; }}
.kpi-pos   {{ background:#16a34a; }}
.kpi-neu   {{ background:#facc15; color:black; }}
.kpi-neg   {{ background:#dc2626; }}
</style>
""", unsafe_allow_html=True)

# =====================================================
# 11. HEADER
# =====================================================
st.set_page_config(page_title="Türkçe Duygu Analizi", layout="wide")
st.markdown("""
<div class="header-card" style="
text-align:center;
padding:30px;
border-radius:20px;
background: linear-gradient(135deg, #0284c7, #22d3ee, #06b6d4);
color:white;
margin-bottom:20px;">
<h1>📊 Sosyal Medya & Haber Türkçe Duygu Analizi</h1>
<p>Twitter · YouTube · Haber · BERT · Redis Cache</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# 12. INPUT
# =====================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
main_platform = st.selectbox("🌍 Ana Platform", ["Sosyal Medya", "Haber"])
if main_platform == "Sosyal Medya":
    platform = st.selectbox("📱 Sosyal Medya", ["Twitter", "YouTube"])
keyword = st.text_input("🔎 Konu / Keyword", "İNÖNÜ ÜNİVERSİTESİ")
limit = st.slider("📌 Veri Sayısı", 5, 100, 20)
st.markdown('</div>', unsafe_allow_html=True)

redis_key = f"{main_platform.lower()}:{keyword}"

# =====================================================
# 13. BUTTONS
# =====================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
b1, b2, b3 = st.columns(3)
fetch = b1.button("📥 Veriyi Çek")
show = b2.button("📤 Redis Verileri")
analyze = b3.button("🤖 Analiz Et")
st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# 14. FETCH
# =====================================================
if fetch:
    data = redis_client.lrange(redis_key, 0, -1)
    if not data:
        if main_platform == "Sosyal Medya":
            data = fetch_tweets(keyword, limit) if platform=="Twitter" else fetch_youtube_comments(keyword, limit)
        else:
            data = fetch_news(keyword, limit)
        save_to_redis(redis_key, data)

    if main_platform == "Haber":
        df_news = load_news_from_redis(redis_key)
        if not df_news.empty:
            df_news["Link"] = df_news.apply(lambda r: f"[{r['Başlık']}]({r['URL']})", axis=1)
            st.table(df_news[["Link"]])
        else:
            st.info("Haber bulunamadı.")
    else:
        st.dataframe(pd.DataFrame({"text": data}), use_container_width=True)

# =====================================================
# 15. SHOW
# =====================================================
if show:
    if main_platform == "Haber":
        df_news = load_news_from_redis(redis_key)
        if not df_news.empty:
            df_news["Link"] = df_news.apply(lambda r: f"[{r['Başlık']}]({r['URL']})", axis=1)
            st.table(df_news[["Link"]])
        else:
            st.info("Haber bulunamadı.")
    else:
        data = redis_client.lrange(redis_key, 0, -1)
        st.dataframe(pd.DataFrame({"text": data}), use_container_width=True)

# =====================================================
# 16. ANALYZE
# =====================================================
if analyze:
    if main_platform == "Haber":
        df = load_news_from_redis(redis_key)
        if df.empty:
            st.warning("Haber verisi yok")
            st.stop()
        df["clean"] = df["Başlık"].apply(clean_text)
        df["Duygu"] = df["clean"].apply(predict_sentiment)
    else:
        data = redis_client.lrange(redis_key, 0, -1)
        df = pd.DataFrame({"text": data})
        df["clean"] = df["text"].apply(clean_text)
        df["Duygu"] = df["clean"].apply(predict_sentiment)

    # KPI ve Grafik
    counts = df["Duygu"].value_counts()
    total = len(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='kpi-box kpi-total'><h2>{total}</h2><p>Toplam</p></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='kpi-box kpi-pos'><h2>{counts.get('Olumlu',0)}</h2><p>Olumlu</p></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='kpi-box kpi-neu'><h2>{counts.get('Nötr',0)}</h2><p>Nötr</p></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='kpi-box kpi-neg'><h2>{counts.get('Olumsuz',0)}</h2><p>Olumsuz</p></div>", unsafe_allow_html=True)

    left, right = st.columns([2,1])
    with left:
        if main_platform == "Haber":
            df_display = df.copy()
            df_display["Link"] = df_display.apply(lambda r: f"[{r['Başlık']}]({r['URL']})", axis=1)
            st.table(df_display[["Link","Duygu"]])
        else:
            st.dataframe(df, use_container_width=True)

    with right:
        fig, ax = plt.subplots()
        ax.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=["#dc2626","#facc15","#16a34a"],
            wedgeprops={"width":0.4}
        )
        ax.axis("equal")
        st.pyplot(fig)

    # CSV & Excel
    if main_platform == "Haber":
        st.download_button("⬇️ CSV İndir", df[["Başlık","URL","Duygu"]].to_csv(index=False), "analiz.csv")
        buffer = BytesIO()
        df[["Başlık","URL","Duygu"]].to_excel(buffer, index=False, engine="openpyxl")
        buffer.seek(0)
        st.download_button("⬇️ Excel İndir", buffer, "analiz.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.download_button("⬇️ CSV İndir", df[["text","Duygu"]].to_csv(index=False), "analiz.csv")
        buffer = BytesIO()
        df[["text","Duygu"]].to_excel(buffer, index=False, engine="openpyxl")
        buffer.seek(0)
        st.download_button("⬇️ Excel İndir", buffer, "analiz.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
