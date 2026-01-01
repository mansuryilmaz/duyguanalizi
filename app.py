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
from dotenv import load_dotenv
import redis
from io import BytesIO
from googleapiclient.discovery import build

# =====================================================
# 0. ENV
# =====================================================
load_dotenv()

BEARER_TOKEN = os.getenv("BEARER_TOKEN")
YT_API_KEY = os.getenv("YT_API_KEY")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# =====================================================
# 1. REDIS
# =====================================================
redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
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
MODEL = "savasy/bert-base-turkish-sentiment-cased"

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
# 5. CLEAN
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
# 6. PREDICT
# =====================================================
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    return labels[torch.argmax(probs).item()]

# =====================================================
# 7. FETCH TWITTER
# =====================================================
@st.cache_data(ttl=600)
def fetch_tweets(keyword, count):
    tweets = []
    res = twitter_client.search_recent_tweets(
        query=f"{keyword} lang:tr -is:retweet",
        max_results=min(count, 100)
    )
    if res.data:
        tweets = [t.text for t in res.data]
    return tweets

# =====================================================
# 8. FETCH YOUTUBE
# =====================================================
def get_video_ids(keyword, max_results=5):
    req = youtube.search().list(
        q=keyword,
        part="id",
        type="video",
        maxResults=max_results
    )
    res = req.execute()
    return [item["id"]["videoId"] for item in res["items"]]

@st.cache_data(ttl=600)
def fetch_youtube_comments(keyword, limit):
    comments = []
    for vid in get_video_ids(keyword):
        req = youtube.commentThreads().list(
            part="snippet",
            videoId=vid,
            maxResults=100,
            textFormat="plainText"
        )
        res = req.execute()
        for item in res["items"]:
            comments.append(item["snippet"]["topLevelComment"]["snippet"]["textDisplay"])
            if len(comments) >= limit:
                return comments
    return comments

# =====================================================
# 9. TEMA + KPI CSS
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
.kpi-box h2 {{ font-size:22px; margin:0; }}
.kpi-box p {{ font-size:13px; margin:0; opacity:0.9; }}

.kpi-total {{ background:#2563eb; }}
.kpi-pos   {{ background:#16a34a; }}
.kpi-neu   {{ background:#facc15; color:black; }}
.kpi-neg   {{ background:#dc2626; }}
</style>
""", unsafe_allow_html=True)

# =====================================================
# 10. HEADER (ORTALANMIŞ & ANİMASYONLU TASARIM)
# =====================================================
st.set_page_config(page_title="Türkçe Duygu Analizi", layout="wide")

st.markdown("""
<style>
@keyframes gradientShift {
    0% {background-position:0% 50%;}
    50% {background-position:100% 50%;}
    100% {background-position:0% 50%;}
}

.header-card {
    text-align:center;
    padding:30px;
    border-radius:20px;
    background: linear-gradient(135deg, #0284c7, #22d3ee, #06b6d4);
    background-size: 300% 300%;
    color:white;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    margin-bottom:20px;
    animation: gradientShift 8s ease infinite;
}

.header-card h1 {
    font-size:36px;
    font-weight:800;
    margin-bottom:10px;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.3);
    transition: transform 0.3s;
}

.header-card h1:hover {
    transform: scale(1.05) rotate(1deg);
}

.header-card p {
    font-size:16px;
    margin:0;
    opacity:0.9;
    transition: opacity 0.3s;
}

.header-card p:hover {
    opacity:1;
}
</style>

<div class="header-card">
    <h1>📊 Sosyal Medya Türkçe Duygu Analizi</h1>
    <p>Twitter · YouTube · BERT · Redis Cache</p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 11. INPUT
# =====================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
platform = st.selectbox("🌍 Platform", ["Twitter", "YouTube"])
keyword = st.text_input("🔎 Konu / Hashtag", "İNÖNÜ ÜNİVERSİTESİ")
limit = st.slider("📌 Veri Sayısı", 10, 100, 30)
st.markdown('</div>', unsafe_allow_html=True)

redis_key = f"{platform.lower()}:{keyword}"

# =====================================================
# 12. BUTTONS
# =====================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
b1, b2, b3 = st.columns(3)
fetch = b1.button("📥 Veriyi Çek")
show = b2.button("📤 Redis Verileri")
analyze = b3.button("🤖 Analiz Et")
st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# 13. FETCH
# =====================================================
if fetch:
    data = redis_client.lrange(redis_key, 0, -1)
    if not data:
        data = fetch_tweets(keyword, limit) if platform == "Twitter" else fetch_youtube_comments(keyword, limit)
        redis_client.delete(redis_key)
        for d in data:
            redis_client.lpush(redis_key, d)
        redis_client.expire(redis_key, 600)
    st.dataframe(pd.DataFrame({"text": data}), use_container_width=True)

# =====================================================
# 14. SHOW
# =====================================================
if show:
    data = redis_client.lrange(redis_key, 0, -1)
    st.dataframe(pd.DataFrame({"text": data}), use_container_width=True)

# =====================================================
# 15. ANALYZE
# =====================================================
if analyze:
    data = redis_client.lrange(redis_key, 0, -1)
    if not data:
        st.warning("Redis’te veri yok")
        st.stop()

    df = pd.DataFrame({"text": data})
    df["clean"] = df["text"].apply(clean_text)
    df["Duygu"] = df["clean"].apply(predict_sentiment)

    counts = df["Duygu"].value_counts()
    total = len(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='kpi-box kpi-total'><p>Toplam</p><h2>{total}</h2></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='kpi-box kpi-pos'><p>Olumlu</p><h2>{counts.get('Olumlu',0)}</h2></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='kpi-box kpi-neu'><p>Nötr</p><h2>{counts.get('Nötr',0)}</h2></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='kpi-box kpi-neg'><p>Olumsuz</p><h2>{counts.get('Olumsuz',0)}</h2></div>", unsafe_allow_html=True)

    left, right = st.columns([2, 1])

    with left:
        st.dataframe(df, use_container_width=True)

    with right:
        fig, ax = plt.subplots()
        labels_pie = counts.index.tolist()
        sizes = counts.values.tolist()

        colors = ["#dc2626", "#facc15", "#16a34a"]

        ax.pie(
            sizes,
            labels=labels_pie,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            wedgeprops={"width": 0.4}
        )
        ax.set_title("Duygu Dağılımı")
        ax.axis("equal")
        st.pyplot(fig)

    st.download_button("⬇️ CSV İndir", df.to_csv(index=False), "analiz.csv", "text/csv")

    buffer = BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    buffer.seek(0)

    st.download_button(
        "⬇️ Excel İndir",
        buffer,
        "analiz.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
