"""
NLPユーティリティ
レビュー要約（軽量モード / Transformersオプション）
"""
import re
from typing import List, Optional
from collections import Counter

# Transformers関連のインポート（オプション）
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


def extract_keywords_lightweight(text: str, top_n: int = 10) -> List[str]:
    """
    軽量モード：キーワード抽出（単純な頻度ベース）
    
    Args:
        text: レビューテキスト
        top_n: 上位N個のキーワードを返す
    
    Returns:
        キーワードリスト
    """
    if not text:
        return []
    
    # ストップワード（簡易版）
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',
        'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'hotel', 'room', 'stay', 'stayed', 'very', 'really', 'quite', 'just',
        'also', 'only', 'not', 'no', 'yes', 'good', 'bad', 'nice', 'great'
    }
    
    # テキストを正規化
    text_lower = text.lower()
    
    # 単語を抽出（英数字のみ、2文字以上）
    words = re.findall(r'\b[a-z]{2,}\b', text_lower)
    
    # ストップワードを除外
    words = [w for w in words if w not in stopwords]
    
    # 頻度カウント
    word_freq = Counter(words)
    
    # 上位N個を返す
    top_words = [word for word, _ in word_freq.most_common(top_n)]
    
    return top_words


def summarize_lightweight(text: str, max_sentences: int = 3) -> str:
    """
    軽量モード：簡易要約（最初のN文を返す）
    
    Args:
        text: レビューテキスト
        max_sentences: 最大文数
    
    Returns:
        要約テキスト
    """
    if not text:
        return ""
    
    # 文を分割（簡易版：. ! ? で分割）
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 最初のN文を結合
    summary = '. '.join(sentences[:max_sentences])
    if summary and not summary.endswith('.'):
        summary += '.'
    
    return summary


def analyze_sentiment_lightweight(text: str) -> str:
    """
    軽量モード：簡易感情分析（キーワードベース）
    
    Args:
        text: レビューテキスト
    
    Returns:
        "positive", "negative", "neutral"
    """
    if not text:
        return "neutral"
    
    text_lower = text.lower()
    
    # ポジティブキーワード
    positive_words = [
        'good', 'great', 'excellent', 'wonderful', 'amazing', 'fantastic',
        'perfect', 'love', 'loved', 'enjoyed', 'satisfied', 'happy', 'nice',
        'clean', 'comfortable', 'convenient', 'helpful', 'friendly', 'beautiful'
    ]
    
    # ネガティブキーワード
    negative_words = [
        'bad', 'terrible', 'awful', 'horrible', 'disappointed', 'disappointing',
        'dirty', 'noisy', 'uncomfortable', 'rude', 'unhelpful', 'poor', 'worst',
        'problem', 'issues', 'complaint', 'unacceptable', 'broken', 'smell'
    ]
    
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        return "positive"
    elif negative_count > positive_count:
        return "negative"
    else:
        return "neutral"


def summarize_with_transformers(text: str) -> Optional[str]:
    """
    Transformersモード：要約（BART等を使用）
    
    Args:
        text: レビューテキスト
    
    Returns:
        要約テキスト、エラー時はNone
    """
    if not TRANSFORMERS_AVAILABLE:
        return None
    
    try:
        # 要約パイプライン（初回実行時にモデルをダウンロード）
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        
        # テキストが長すぎる場合は切り詰め
        max_length = 1024
        if len(text) > max_length:
            text = text[:max_length]
        
        result = summarizer(text, max_length=100, min_length=30, do_sample=False)
        return result[0]['summary_text']
        
    except Exception as e:
        print(f"Transformers summarization error: {str(e)}")
        return None


def analyze_sentiment_with_transformers(text: str) -> Optional[dict]:
    """
    Transformersモード：感情分析
    
    Args:
        text: レビューテキスト
    
    Returns:
        {"label": ..., "score": ...} または None
    """
    if not TRANSFORMERS_AVAILABLE:
        return None
    
    try:
        # 感情分析パイプライン
        classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        
        # テキストが長すぎる場合は切り詰め
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]
        
        result = classifier(text)
        return {
            "label": result[0]['label'],
            "score": result[0]['score']
        }
        
    except Exception as e:
        print(f"Transformers sentiment analysis error: {str(e)}")
        return None


def process_reviews(
    review_text: str,
    mode: str = "lightweight"
) -> dict:
    """
    レビューテキストを処理（要約・キーワード・感情分析）
    
    Args:
        review_text: レビューテキスト
        mode: "lightweight" または "transformers"
    
    Returns:
        処理結果のdict
    """
    if not review_text:
        return {
            "summary": "",
            "keywords": [],
            "sentiment": "neutral"
        }
    
    result = {}
    
    if mode == "transformers" and TRANSFORMERS_AVAILABLE:
        # Transformersモード
        summary = summarize_with_transformers(review_text)
        sentiment_result = analyze_sentiment_with_transformers(review_text)
        
        result["summary"] = summary or summarize_lightweight(review_text)
        result["keywords"] = extract_keywords_lightweight(review_text)
        result["sentiment"] = sentiment_result.get("label", "neutral") if sentiment_result else "neutral"
        result["sentiment_score"] = sentiment_result.get("score", 0.0) if sentiment_result else 0.0
    else:
        # 軽量モード
        result["summary"] = summarize_lightweight(review_text)
        result["keywords"] = extract_keywords_lightweight(review_text)
        result["sentiment"] = analyze_sentiment_lightweight(review_text)
        result["sentiment_score"] = 0.0
    
    return result

