"""
APIクライアント
Google Places API, OSM Overpass API の呼び出し
"""
import time
import requests
from typing import List, Dict, Optional
import config
import utils


def fetch_google_places_hotels(city: str, max_results: int = 30) -> List[Dict]:
    """
    Google Places APIでホテルを検索
    
    Args:
        city: 都市名
        max_results: 最大取得件数
    
    Returns:
        ホテル情報のリスト
    """
    if not config.GOOGLE_PLACES_API_KEY:
        raise ValueError(config.ERROR_MISSING_API_KEY + " (GOOGLE_PLACES_API_KEY)")
    
    hotels = []
    query = f"{city} hotel"
    
    # Text Search APIを使用
    url = f"{config.GOOGLE_PLACES_BASE_URL}/textsearch/json"
    params = {
        "query": query,
        "key": config.GOOGLE_PLACES_API_KEY,
        "type": "lodging"  # 宿泊施設に限定
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") != "OK":
            error_msg = data.get("error_message", "Unknown error")
            status = data.get("status", "UNKNOWN")
            
            # より分かりやすいエラーメッセージ
            if status == "REQUEST_DENIED":
                detailed_msg = (
                    f"Google Places API が拒否されました。\n"
                    f"以下の手順を確認してください：\n"
                    f"1. Google Cloud Console (https://console.cloud.google.com/) にアクセス\n"
                    f"2. プロジェクトを選択\n"
                    f"3. 「APIとサービス」→「ライブラリ」から「Places API」を検索\n"
                    f"4. 「Places API」を有効化\n"
                    f"5. 「認証情報」でAPIキーの制限を確認（必要に応じて「Places API」を許可）\n"
                    f"エラー詳細: {error_msg}"
                )
                raise ValueError(detailed_msg)
            else:
                raise ValueError(f"Google Places API error: {status} - {error_msg}")
        
        results = data.get("results", [])
        
        for result in results[:max_results]:
            geometry = result.get("geometry", {})
            location = geometry.get("location", {})
            
            hotel = {
                "place_id": result.get("place_id", ""),
                "name": result.get("name", ""),
                "lat": location.get("lat", 0.0),
                "lon": location.get("lng", 0.0),
                "rating": result.get("rating", 0.0),
                "user_ratings_total": result.get("user_ratings_total", 0),
                "address": result.get("formatted_address", ""),
                "types": result.get("types", [])
            }
            hotels.append(hotel)
        
        # レート制限対策
        time.sleep(0.1)
        
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Google Places API request failed: {str(e)}")
    
    return hotels


def fetch_attractions_google(
    city_name: str,
    center_lat: float = None,
    center_lng: float = None,
    radius_m: int = 10000,
    max_results: int = 15
) -> List[Dict]:
    """
    Google Places APIで観光地を取得
    
    Args:
        city_name: 都市名
        center_lat, center_lng: 中心地点の緯度・経度（オプション、指定しない場合はcity_nameから取得）
        radius_m: 検索半径（メートル、Nearby Search使用時）
        max_results: 最大取得件数
    
    Returns:
        観光地情報のリスト（既存のデータ構造と互換性を保つ）
    """
    if not config.GOOGLE_PLACES_API_KEY:
        raise ValueError(config.ERROR_MISSING_API_KEY + " (GOOGLE_PLACES_API_KEY)")
    
    attractions = []
    
    # 複数のクエリで観光地を検索
    queries = [
        f"{city_name} tourist attractions",
        f"{city_name} things to do",
        f"{city_name} landmarks",
        f"{city_name} sightseeing"
    ]
    
    all_results = []
    seen_place_ids = set()
    
    try:
        for query in queries:
            if len(all_results) >= max_results:
                break
            
            # Text Search APIを使用
            url = f"{config.GOOGLE_PLACES_BASE_URL}/textsearch/json"
            params = {
                "query": query,
                "key": config.GOOGLE_PLACES_API_KEY,
                "type": "tourist_attraction"  # 観光地に限定
            }
            
            try:
                response = requests.get(url, params=params, timeout=10)
                
                # HTTPステータスコードチェック
                if response.status_code == 429:
                    # レート制限エラー
                    print(f"Google Places API rate limit exceeded. Waiting...")
                    time.sleep(2)
                    continue
                elif response.status_code >= 500:
                    # サーバーエラー
                    print(f"Google Places API server error: {response.status_code}")
                    time.sleep(1)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                if data.get("status") == "OK":
                    results = data.get("results", [])
                    for result in results:
                        place_id = result.get("place_id", "")
                        # 重複を避ける
                        if place_id and place_id not in seen_place_ids:
                            seen_place_ids.add(place_id)
                            all_results.append(result)
                            if len(all_results) >= max_results:
                                break
                elif data.get("status") == "REQUEST_DENIED":
                    error_msg = data.get("error_message", "Unknown error")
                    raise ValueError(
                        f"Google Places API が拒否されました: {error_msg}\n"
                        f"Places APIが有効化されているか確認してください。"
                    )
                # その他のエラー（ZERO_RESULTS等）は無視して次のクエリを試す
                
            except requests.exceptions.RequestException as e:
                # ネットワークエラーなどはログに記録して続行
                print(f"Request error for query '{query}': {str(e)}")
                continue
            
            # レート制限対策
            time.sleep(0.2)
        
        # 評価とレビュー数でソート（人気順）
        all_results.sort(
            key=lambda x: (
                x.get("rating", 0) * x.get("user_ratings_total", 0),
                x.get("user_ratings_total", 0)
            ),
            reverse=True
        )
        
        # max_results件に制限
        for result in all_results[:max_results]:
            geometry = result.get("geometry", {})
            location = geometry.get("location", {})
            types = result.get("types", [])
            
            # typesから観光地関連のキーワードを抽出（kindsの代わり）
            relevant_types = [t for t in types if any(keyword in t for keyword in [
                "tourist", "attraction", "museum", "park", "landmark", 
                "shrine", "temple", "monument", "gallery", "zoo"
            ])]
            kinds_str = ", ".join(relevant_types[:3]) if relevant_types else "tourist_attraction"
            
            attraction = {
                "place_id": result.get("place_id", ""),
                "name": result.get("name", "Unknown"),
                "lat": location.get("lat", 0.0),
                "lon": location.get("lng", 0.0),
                "rating": result.get("rating", 0.0),
                "user_ratings_total": result.get("user_ratings_total", 0),
                "address": result.get("formatted_address", ""),
                "types": types,
                "kinds": kinds_str,  # 既存コードとの互換性のため
                "xid": result.get("place_id", "")  # 既存コードとの互換性のため
            }
            attractions.append(attraction)
        
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Google Places API request failed: {str(e)}")
    except Exception as e:
        raise ValueError(f"観光地取得エラー: {str(e)}")
    
    return attractions


def fetch_osm_stations_and_stops(
    lat: float,
    lon: float,
    radius_km: float = 1.0
) -> Dict[str, List[Dict]]:
    """
    OSM Overpass APIで駅とバス停を取得
    
    Args:
        lat, lon: 中心地点の緯度・経度
        radius_km: 検索半径（km）
    
    Returns:
        {"stations": [...], "bus_stops": [...]} の形式
    """
    # Overpass QLクエリを作成
    # 半径を度に変換（簡易版：1度 ≈ 111km）
    radius_deg = radius_km / 111.0
    
    query = f"""
    [out:json][timeout:25];
    (
      node["railway"="station"](around:{radius_km * 1000},{lat},{lon});
      way["railway"="station"](around:{radius_km * 1000},{lat},{lon});
      node["highway"="bus_stop"](around:{radius_km * 1000},{lat},{lon});
    );
    out center;
    """
    
    try:
        response = requests.post(
            config.OSM_OVERPASS_URL,
            data={"data": query},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        stations = []
        bus_stops = []
        
        elements = data.get("elements", [])
        
        for element in elements:
            if element.get("type") == "node":
                lat_elem = element.get("lat", 0.0)
                lon_elem = element.get("lon", 0.0)
            elif element.get("type") == "way" and "center" in element:
                lat_elem = element["center"].get("lat", 0.0)
                lon_elem = element["center"].get("lon", 0.0)
            else:
                continue
            
            tags = element.get("tags", {})
            name = tags.get("name", "Unknown")
            
            point = {
                "name": name,
                "lat": lat_elem,
                "lon": lon_elem,
                "type": element.get("type", "")
            }
            
            if tags.get("railway") == "station":
                stations.append(point)
            elif tags.get("highway") == "bus_stop":
                bus_stops.append(point)
        
        time.sleep(0.5)  # Overpass APIのレート制限対策
        
    except requests.exceptions.RequestException as e:
        print(f"OSM Overpass API error: {str(e)}")
        return {"stations": [], "bus_stops": []}
    
    return {
        "stations": stations,
        "bus_stops": bus_stops
    }


def get_city_coordinates(city: str) -> Optional[Dict[str, float]]:
    """
    都市名から緯度・経度を取得（Google Geocoding APIを使用）
    
    Args:
        city: 都市名
    
    Returns:
        {"lat": ..., "lon": ...} または None
    """
    if not config.GOOGLE_PLACES_API_KEY:
        return None
    
    url = f"{config.GOOGLE_PLACES_BASE_URL}/geocode/json"
    params = {
        "address": city,
        "key": config.GOOGLE_PLACES_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") == "OK" and data.get("results"):
            location = data["results"][0]["geometry"]["location"]
            return {
                "lat": location.get("lat", 0.0),
                "lon": location.get("lng", 0.0)
            }
    except Exception as e:
        print(f"Geocoding error: {str(e)}")
    
    return None

