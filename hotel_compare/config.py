"""
設定管理モジュール
環境変数からAPIキーなどを読み込む
"""
import os
from dotenv import load_dotenv

# .envファイルを読み込む
load_dotenv()

# APIキー設定
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")

# API設定
GOOGLE_PLACES_BASE_URL = "https://maps.googleapis.com/maps/api/place"
OSM_OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# デフォルト設定
DEFAULT_HOTEL_COUNT = 30
DEFAULT_ATTRACTION_COUNT = 15
DEFAULT_SEARCH_RADIUS_KM = 1.0  # 駅・バス停検索の半径（km）
DEFAULT_FUZZY_MATCH_THRESHOLD = 70  # fuzzy matchingの閾値（0-100）

# エラーメッセージ
ERROR_MISSING_API_KEY = "APIキーが設定されていません。.envファイルを確認してください。"

