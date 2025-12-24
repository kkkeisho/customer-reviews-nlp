# app.py
# Streamlit demo: 100 dummy hotel reviews -> aspect (multi-label) + sentiment -> dashboard
#
# Run:
#   pip install streamlit pandas numpy
#   # Optional (heavy): pip install transformers torch
#   streamlit run app.py
#
# Notes:
# - Default mode uses a lightweight keyword-based classifier (fast, no model download).
# - Optional mode uses HuggingFace zero-shot (BART MNLI) + sentiment pipeline (can be slow / needs model download).

import re
import random
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------
# 1) Config
# ----------------------------
ASPECTS = [
    "cleanliness",
    "staff",
    "facility",
    "noise",
    "location",
    "price/value",
    "food",
    "checkin/checkout",
    "wifi",
    "other",
]

# Keyword dictionary for lightweight aspect detection (edit freely)
ASPECT_KEYWORDS = {
    "cleanliness": ["clean", "dirty", "smell", "stain", "hair", "bathroom", "mold", "dust", "towel"],
    "staff": ["staff", "rude", "friendly", "helpful", "unhelpful", "service", "receptionist", "manager"],
    "facility": ["room", "bed", "aircon", "ac", "heater", "shower", "elevator", "gym", "pool", "facility", "amenities"],
    "noise": ["noisy", "noise", "loud", "thin walls", "neighbors", "street noise", "construction"],
    "location": ["location", "near", "far", "walk", "subway", "station", "central", "downtown", "airport"],
    "price/value": ["price", "value", "expensive", "cheap", "worth", "overpriced", "cost"],
    "food": ["breakfast", "food", "restaurant", "coffee", "buffet", "dinner"],
    "checkin/checkout": ["check-in", "checkin", "check out", "checkout", "queue", "line", "waiting", "front desk"],
    "wifi": ["wifi", "wi-fi", "internet", "connection", "signal"],
}

# Keyword dictionary for lightweight sentiment
POS_WORDS = ["great", "good", "excellent", "amazing", "clean", "comfortable", "friendly", "helpful", "fast", "quiet"]
NEG_WORDS = ["bad", "terrible", "awful", "dirty", "rude", "slow", "noisy", "smell", "broken", "worst", "forever"]

SOURCES = ["Booking.com", "Expedia", "Google", "Agoda", "Tripadvisor"]
LANGS = ["en", "en", "en", "en", "ja"]  # mostly English, some Japanese for realism


# ----------------------------
# 2) Dummy review generator
# ----------------------------
@dataclass
class DummyReview:
    hotel_id: str
    source: str
    lang: str
    date: datetime
    rating: int
    text: str


def _pick_phrase(aspect: str, sentiment: str) -> str:
    # short phrase templates per aspect
    templates = {
        "cleanliness": {
            "pos": ["The room was very clean.", "Bathroom was spotless.", "Fresh towels and no smell."],
            "neg": ["The room was not clean.", "There was a bad smell in the bathroom.", "Found hair on the bed."],
        },
        "staff": {
            "pos": ["Staff were friendly and helpful.", "Reception was very kind.", "Service was excellent."],
            "neg": ["Staff was rude.", "Receptionist was unhelpful.", "Customer service was terrible."],
        },
        "facility": {
            "pos": ["Bed was comfortable.", "Facilities were great.", "Shower pressure was good."],
            "neg": ["Aircon was broken.", "Elevator was slow and old.", "The room felt outdated."],
        },
        "noise": {
            "pos": ["It was quiet at night.", "No noise issues.", "Soundproofing was decent."],
            "neg": ["Very noisy at night.", "Thin walls, could hear neighbors.", "Street noise was loud."],
        },
        "location": {
            "pos": ["Great location near the station.", "Easy walk to downtown.", "Convenient area."],
            "neg": ["Location was far from everything.", "Hard to access without a car.", "Not a convenient area."],
        },
        "price/value": {
            "pos": ["Good value for the price.", "Worth the cost.", "Reasonably priced."],
            "neg": ["Overpriced for what you get.", "Not worth the price.", "Too expensive."],
        },
        "food": {
            "pos": ["Breakfast was tasty.", "Good buffet options.", "Nice coffee and food."],
            "neg": ["Breakfast was disappointing.", "Food quality was bad.", "Limited options at the restaurant."],
        },
        "checkin/checkout": {
            "pos": ["Check-in was smooth and quick.", "No waiting at the front desk.", "Checkout was easy."],
            "neg": ["The check-in took forever.", "Long queue at reception.", "Checkout process was slow."],
        },
        "wifi": {
            "pos": ["WiFi was fast and stable.", "Internet worked perfectly.", "Strong signal in the room."],
            "neg": ["WiFi was terrible.", "Internet kept disconnecting.", "Weak signal in the room."],
        },
        "other": {
            "pos": ["Overall a great stay.", "Would come again.", "Loved the experience."],
            "neg": ["Overall disappointing.", "Would not stay again.", "Not recommended."],
        },
    }
    return random.choice(templates.get(aspect, templates["other"])[sentiment])


def generate_dummy_reviews(n: int = 100, seed: int = 42) -> list[DummyReview]:
    random.seed(seed)
    np.random.seed(seed)

    hotels = [f"H{str(i).zfill(3)}" for i in range(1, 6)]  # 5 hotels
    start = datetime.now() - timedelta(days=90)

    reviews: list[DummyReview] = []
    for _ in range(n):
        hotel_id = random.choice(hotels)
        source = random.choice(SOURCES)
        lang = random.choice(LANGS)
        date = start + timedelta(days=int(np.random.randint(0, 90)))

        # Choose 1-3 aspects
        k = int(np.random.choice([1, 2, 3], p=[0.45, 0.40, 0.15]))
        aspects = random.sample(ASPECTS[:-1], k=k)  # exclude "other" for selection; we can add later
        # Decide overall sentiment bias
        overall = np.random.choice(["pos", "neg"], p=[0.62, 0.38])

        # Create text by combining aspect phrases + occasional "other"
        phrases = [_pick_phrase(a, overall) for a in aspects]
        if np.random.rand() < 0.25:
            phrases.append(_pick_phrase("other", overall))

        text_en = " ".join(phrases)

        # Quick Japanese flavor (simple)
        if lang == "ja":
            text = text_en.replace("Staff was rude.", "スタッフが失礼でした。").replace(
                "The room was very clean.", "部屋はとても清潔でした。"
            )
        else:
            text = text_en

        # Rating correlated with overall
        rating = int(np.random.choice([5, 4, 3, 2, 1], p=[0.35, 0.27, 0.18, 0.12, 0.08])) if overall == "pos" else int(
            np.random.choice([5, 4, 3, 2, 1], p=[0.05, 0.10, 0.20, 0.30, 0.35])
        )

        reviews.append(DummyReview(hotel_id, source, lang, date, rating, text))
    return reviews


# ----------------------------
# 3) Lightweight classifiers
# ----------------------------
def keyword_aspect_scores(text: str, labels: list[str]) -> dict[str, float]:
    t = text.lower()
    scores = {}
    for lab in labels:
        if lab == "other":
            continue
        hits = 0
        for kw in ASPECT_KEYWORDS.get(lab, []):
            if kw in t:
                hits += 1
        # soft score: diminishing returns
        scores[lab] = 1 - np.exp(-hits / 2)  # 0..~1
    # other score: inverse of max hit (if nothing matches => higher "other")
    max_hit = max(scores.values()) if scores else 0.0
    scores["other"] = float(max(0.0, 0.6 - max_hit))
    return scores


def keyword_sentiment(text: str) -> tuple[str, float]:
    t = text.lower()
    pos = sum(1 for w in POS_WORDS if w in t)
    neg = sum(1 for w in NEG_WORDS if w in t)
    if pos == 0 and neg == 0:
        return ("NEUTRAL", 0.50)
    if neg > pos:
        conf = min(0.55 + (neg - pos) * 0.10, 0.99)
        return ("NEGATIVE", float(conf))
    if pos > neg:
        conf = min(0.55 + (pos - neg) * 0.10, 0.99)
        return ("POSITIVE", float(conf))
    return ("NEUTRAL", 0.55)


# ----------------------------
# 4) Optional Transformers pipelines
# ----------------------------
@st.cache_resource
def load_transformers_pipelines():
    from transformers import pipeline

    zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    sent = pipeline("sentiment-analysis")  # default model
    return zs, sent


def classify_review(
    text: str,
    labels: list[str],
    mode: str,
    top_k: int,
    min_score: float,
) -> dict:
    """
    Returns:
      {
        "sentiment_label": ...,
        "sentiment_score": ...,
        "labels": [..sorted..],
        "scores": [..],
        "selected": [(label, score), ...]
      }
    """
    if mode == "Transformers (zero-shot + sentiment)":
        zs, sent = load_transformers_pipelines()
        out_zs = zs(text, labels)
        out_sent = sent(text)[0]
        ranked = list(zip(out_zs["labels"], out_zs["scores"]))
    else:
        # lightweight
        aspect_scores = keyword_aspect_scores(text, labels)
        ranked = sorted(aspect_scores.items(), key=lambda x: x[1], reverse=True)
        s_lab, s_score = keyword_sentiment(text)
        out_sent = {"label": s_lab, "score": s_score}

    selected = [(lab, float(sc)) for lab, sc in ranked[:top_k] if sc >= min_score]

    return {
        "sentiment_label": out_sent["label"],
        "sentiment_score": float(out_sent["score"]),
        "labels": [lab for lab, _ in ranked],
        "scores": [float(sc) for _, sc in ranked],
        "selected": selected,
    }


def to_aspect_events(df_reviews: pd.DataFrame, col_selected: str = "selected") -> pd.DataFrame:
    rows = []
    for _, r in df_reviews.iterrows():
        for (aspect, conf) in r[col_selected]:
            rows.append(
                {
                    "hotel_id": r["hotel_id"],
                    "review_id": r["review_id"],
                    "date": r["date"],
                    "source": r["source"],
                    "lang": r["lang"],
                    "rating": r["rating"],
                    "aspect": aspect,
                    "aspect_confidence": float(conf),
                    "sentiment": r["sentiment_label"],
                    "sentiment_confidence": float(r["sentiment_score"]),
                    "text": r["text"],
                }
            )
    return pd.DataFrame(rows)


# ----------------------------
# 5) Streamlit UI
# ----------------------------
st.set_page_config(page_title="Hotel Review NLP Dashboard (Dummy)", layout="wide")

st.title("レビュー仕分け（NLP）デモ：100件ダミーレビュー → アスペクト×感情 → 改善に繋がる可視化")

with st.sidebar:
    st.header("設定")
    seed = st.number_input("乱数シード", min_value=0, max_value=10_000, value=42, step=1)
    n = st.slider("ダミーレビュー件数", 20, 500, 100, 10)

    mode = st.selectbox(
        "分類モード",
        ["Lightweight (keyword)", "Transformers (zero-shot + sentiment)"],
        index=0,
        help="Transformersは初回にモデルDLが必要で重いです。",
    )
    top_k = st.slider("アスペクトTop-K採用", 1, 5, 3, 1)
    min_score = st.slider("アスペクト採用の最低スコア閾値", 0.0, 0.8, 0.10, 0.01)
    st.caption("※ zero-shotのscoreは“確率”ではないので、運用は件数/率で見るのが安全です。")

    st.divider()
    st.subheader("フィルタ")
    hotel_filter = st.multiselect("ホテル", options=[f"H{str(i).zfill(3)}" for i in range(1, 6)], default=[])
    source_filter = st.multiselect("ソース", options=SOURCES, default=[])
    sentiment_filter = st.multiselect("感情", options=["POSITIVE", "NEGATIVE", "NEUTRAL"], default=[])


@st.cache_data
def build_reviews_df(n: int, seed: int) -> pd.DataFrame:
    reviews = generate_dummy_reviews(n=n, seed=seed)
    df = pd.DataFrame(
        [
            {
                "review_id": f"R{idx:04d}",
                "hotel_id": r.hotel_id,
                "source": r.source,
                "lang": r.lang,
                "date": r.date.date(),
                "rating": r.rating,
                "text": r.text,
            }
            for idx, r in enumerate(reviews, start=1)
        ]
    )
    return df


df = build_reviews_df(n=n, seed=seed).copy()

# Apply filters
if hotel_filter:
    df = df[df["hotel_id"].isin(hotel_filter)]
if source_filter:
    df = df[df["source"].isin(source_filter)]

# Classify
with st.spinner("分類中..."):
    results = []
    for text in df["text"].tolist():
        results.append(classify_review(text, ASPECTS, mode, top_k, min_score))

df["sentiment_label"] = [r["sentiment_label"] for r in results]
df["sentiment_score"] = [r["sentiment_score"] for r in results]
df["ranked_labels"] = [r["labels"] for r in results]
df["ranked_scores"] = [r["scores"] for r in results]
df["selected"] = [r["selected"] for r in results]

if sentiment_filter:
    df = df[df["sentiment_label"].isin(sentiment_filter)]

events = to_aspect_events(df)

# ----------------------------
# 6) Dashboard
# ----------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("レビュー件数", int(df.shape[0]))
col2.metric("イベント件数（レビュー×アスペクト）", int(events.shape[0]))
neg_rate = (events["sentiment"].eq("NEGATIVE").mean() * 100) if len(events) else 0
col3.metric("NEG率（イベント基準）", f"{neg_rate:.1f}%")
top_aspect = (
    events[events["sentiment"] == "NEGATIVE"]["aspect"].value_counts().head(1).index[0]
    if len(events) and (events["sentiment"] == "NEGATIVE").any()
    else "-"
)
col4.metric("最多NEGアスペクト", top_aspect)

st.divider()

left, right = st.columns([1, 1])

with left:
    st.subheader("アスペクト別：件数（イベント）")
    if len(events):
        aspect_counts = events["aspect"].value_counts().rename_axis("aspect").reset_index(name="count")
        st.bar_chart(aspect_counts.set_index("aspect"))
    else:
        st.info("データがありません（フィルタ条件を緩めてください）。")

with right:
    st.subheader("アスペクト別：NEG率（イベント）")
    if len(events):
        grp = events.groupby("aspect")["sentiment"].apply(lambda s: (s == "NEGATIVE").mean()).reset_index(name="neg_rate")
        grp["neg_rate"] = grp["neg_rate"] * 100
        st.bar_chart(grp.set_index("aspect"))
    else:
        st.info("データがありません。")

st.divider()

st.subheader("ホテル×アスペクト：NEG件数ヒートマップ（表）")
if len(events):
    pivot = (
        events[events["sentiment"] == "NEGATIVE"]
        .pivot_table(index="hotel_id", columns="aspect", values="review_id", aggfunc="count", fill_value=0)
        .astype(int)
    )
    st.dataframe(pivot, width='stretch')
else:
    st.info("NEGイベントがありません。")

st.divider()

st.subheader("レビュー一覧（分類結果つき）")
show_cols = ["review_id", "hotel_id", "date", "source", "rating", "sentiment_label", "sentiment_score", "selected", "text"]
df_display = df[show_cols].copy()
df_display["selected"] = df_display["selected"].apply(lambda x: str(x) if x else "")
st.dataframe(df_display.sort_values(["date", "hotel_id"], ascending=[False, True]), width='stretch')

st.divider()

st.subheader("改善に直結させる：根拠フレーズ候補（簡易抽出）")
st.caption("※ 本格的にはaspectごとに該当文スパン抽出/要約にすると強い。ここでは雰囲気のデモ。")

if len(events):
    # pick negative events and extract a crude "evidence" snippet
    def evidence_snippet(txt: str, aspect: str) -> str:
        t = txt
        kws = ASPECT_KEYWORDS.get(aspect, [])
        for kw in kws:
            m = re.search(rf"(.{{0,40}}{re.escape(kw)}.{{0,40}})", t, flags=re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return (t[:80] + "…") if len(t) > 80 else t

    neg_events = events[events["sentiment"] == "NEGATIVE"].copy()
    if len(neg_events):
        neg_events["evidence"] = [
            evidence_snippet(t, a) for t, a in zip(neg_events["text"].tolist(), neg_events["aspect"].tolist())
        ]
        # top aspects evidence table
        topA = neg_events["aspect"].value_counts().head(3).index.tolist()
        tab = neg_events[neg_events["aspect"].isin(topA)][
            ["hotel_id", "date", "aspect", "aspect_confidence", "evidence", "source", "rating"]
        ].sort_values(["aspect", "aspect_confidence"], ascending=[True, False])
        st.dataframe(tab.head(50), width='stretch')
    else:
        st.success("NEGイベントがないので、改善アクションは不要（このダミー条件では）")
else:
    st.info("データがありません。")
