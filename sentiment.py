"""
Sentiment Analysis Module for StockSense AI — Indian NSE Edition
- Google News RSS scraping (Indian locale)
- FinBERT sentiment analysis (transformer-based, financial domain)
- Composite Market Sentiment Score (News only)
- Narrative summary generation
"""

import os
# Force transformers to use PyTorch, NOT TensorFlow (avoids conflict with Keras LSTM model)
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import feedparser
import requests
import streamlit as st
from datetime import datetime
from urllib.parse import quote_plus
from config import get_search_term

# ============== FinBERT INITIALISATION ==============

_FINBERT_AVAILABLE = True


@st.cache_resource(show_spinner="Loading FinBERT model (first time only — ~400MB download)...")
def _load_finbert():
    """Load FinBERT sentiment analysis pipeline using PyTorch. Cached across reruns."""
    global _FINBERT_AVAILABLE
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        pipe = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            truncation=True,
            max_length=512,
            framework="pt",
        )
        return pipe
    except Exception as e:
        _FINBERT_AVAILABLE = False
        st.warning(f"⚠️ Could not load FinBERT: {e}. Falling back to VADER.")
        return None


def _get_vader_fallback():
    """Load VADER as fallback if FinBERT fails."""
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()


def _classify_sentiment(compound):
    """Classify compound score into label and emoji."""
    if compound >= 0.05:
        return "Positive", "🟢"
    elif compound <= -0.05:
        return "Negative", "🔴"
    else:
        return "Neutral", "🟡"


# ============== NEWS SCRAPING ==============

@st.cache_data(ttl=900, show_spinner=False)
def fetch_news(ticker, company_name=None, count=10):
    """Fetch top news articles from Google News RSS (Indian locale)."""
    search_term = get_search_term(ticker) if company_name is None else company_name
    locale_params = "hl=en-IN&gl=IN&ceid=IN:en"
    query = quote_plus(f"{search_term} stock NSE")
    url = f"https://news.google.com/rss/search?q={query}&{locale_params}"

    try:
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries[:count]:
            title = entry.get("title", "")
            source = ""
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                title = parts[0]
                source = parts[1]

            published = entry.get("published", "")
            try:
                pub_date = datetime(*entry.published_parsed[:6])
                published_str = pub_date.strftime("%b %d, %Y %I:%M %p")
            except Exception:
                published_str = published

            articles.append({
                "title": title,
                "link": entry.get("link", ""),
                "source": source,
                "published": published_str,
            })
        return articles, None
    except Exception as e:
        return [], f"Could not fetch news: {str(e)}"


# ============== SENTIMENT ANALYSIS (FinBERT) ==============

def analyze_sentiment(texts):
    """Analyze sentiment using FinBERT (financial domain transformer).

    Returns list of dicts: {text, compound, label, emoji, pos, neg, neu,
                            finbert_label, finbert_confidence}
    """
    results = []
    if not texts:
        return results

    finbert = _load_finbert()

    # Fallback to VADER if FinBERT is unavailable
    if finbert is None:
        vader = _get_vader_fallback()
        for text in texts:
            if not text or not text.strip():
                continue
            scores = vader.polarity_scores(text)
            label, emoji = _classify_sentiment(scores["compound"])
            results.append({
                "text": text,
                "compound": scores["compound"],
                "pos": scores["pos"], "neg": scores["neg"], "neu": scores["neu"],
                "label": label, "emoji": emoji,
                "finbert_label": "N/A (VADER fallback)",
                "finbert_confidence": 0.0,
            })
        return results

    # Process with FinBERT
    for text in texts:
        if not text or not text.strip():
            continue
        try:
            text_input = text[:500] if len(text) > 500 else text
            output = finbert(text_input)[0]

            finbert_label = output['label'].lower()
            finbert_score = output['score']

            if finbert_label == 'positive':
                compound = finbert_score
                pos, neg, neu = finbert_score, 0.0, 1 - finbert_score
            elif finbert_label == 'negative':
                compound = -finbert_score
                pos, neg, neu = 0.0, finbert_score, 1 - finbert_score
            else:
                compound = 0.0
                pos, neg, neu = 0.0, 0.0, finbert_score

            label, emoji = _classify_sentiment(compound)

            results.append({
                "text": text, "compound": round(compound, 4),
                "pos": round(pos, 4), "neg": round(neg, 4), "neu": round(neu, 4),
                "label": label, "emoji": emoji,
                "finbert_label": finbert_label,
                "finbert_confidence": round(finbert_score, 4),
            })
        except Exception:
            results.append({
                "text": text, "compound": 0.0,
                "pos": 0.0, "neg": 0.0, "neu": 1.0,
                "label": "Neutral", "emoji": "🟡",
                "finbert_label": "neutral", "finbert_confidence": 0.0,
            })

    return results


def get_aggregate_sentiment(sentiments):
    """Compute aggregate stats from a list of sentiment results."""
    if not sentiments:
        return {
            "avg_compound": 0, "positive_count": 0,
            "negative_count": 0, "neutral_count": 0,
            "total": 0, "label": "No Data", "emoji": "⚪",
        }

    compounds = [s["compound"] for s in sentiments]
    avg = sum(compounds) / len(compounds)
    label, emoji = _classify_sentiment(avg)

    return {
        "avg_compound": avg,
        "positive_count": sum(1 for s in sentiments if s["label"] == "Positive"),
        "negative_count": sum(1 for s in sentiments if s["label"] == "Negative"),
        "neutral_count": sum(1 for s in sentiments if s["label"] == "Neutral"),
        "total": len(sentiments),
        "label": label, "emoji": emoji,
    }


# ============== MARKET SENTIMENT SCORE (News Only) ==============

def compute_market_sentiment(news_sentiments):
    """Compute Market Sentiment Score (0-100) from news articles only.
    Returns (score, label, emoji, news_agg).
    """
    news_agg = get_aggregate_sentiment(news_sentiments)

    if news_agg["total"] == 0:
        return 50, "No Data", "⚪", news_agg

    composite = (news_agg["avg_compound"] + 1) * 50
    composite = max(0, min(100, composite))

    if composite >= 76:
        label, emoji = "Very Bullish", "🟢🟢"
    elif composite >= 56:
        label, emoji = "Bullish", "🟢"
    elif composite >= 46:
        label, emoji = "Neutral", "🟡"
    elif composite >= 26:
        label, emoji = "Bearish", "🔴"
    else:
        label, emoji = "Very Bearish", "🔴🔴"

    return round(composite, 1), label, emoji, news_agg


# ============== NARRATIVE SUMMARY ==============

def generate_sentiment_summary(ticker, news_sentiments, news_agg,
                               sentiment_score, sentiment_label):
    """Generate a human-readable narrative from news sentiment."""
    from config import get_display_name
    name = get_display_name(ticker)
    parts = []

    parts.append(
        f"Market sentiment for {name} ({ticker.replace('.NS', '')}) is currently "
        f"**{sentiment_label}** with a composite score of **{sentiment_score}/100**. "
        f"*(Powered by FinBERT — a transformer model fine-tuned on financial text)*"
    )

    if news_agg["total"] > 0:
        pos = news_agg["positive_count"]
        neg = news_agg["negative_count"]
        total = news_agg["total"]
        parts.append(
            f"Across {total} recent news articles, {pos} are positive and {neg} are negative "
            f"(avg compound: {news_agg['avg_compound']:+.3f})."
        )
        pos_headlines = [s for s in news_sentiments if s["label"] == "Positive"]
        neg_headlines = [s for s in news_sentiments if s["label"] == "Negative"]
        if pos_headlines:
            conf = pos_headlines[0].get('finbert_confidence', 0)
            parts.append(f"Positive signal: *\"{pos_headlines[0]['text'][:80]}...\"* (confidence: {conf:.0%})")
        if neg_headlines:
            conf = neg_headlines[0].get('finbert_confidence', 0)
            parts.append(f"Concern: *\"{neg_headlines[0]['text'][:80]}...\"* (confidence: {conf:.0%})")
    else:
        parts.append("No recent news articles were found for analysis.")

    return " ".join(parts)
