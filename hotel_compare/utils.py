"""
ユーティリティ関数
距離計算など
"""
import math
from typing import Tuple, Optional, List, Dict


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Haversine公式で2点間の距離（km）を計算
    
    Args:
        lat1, lon1: 地点1の緯度・経度
        lat2, lon2: 地点2の緯度・経度
    
    Returns:
        距離（km）
    """
    # 地球の半径（km）
    R = 6371.0
    
    # 度をラジアンに変換
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # 差を計算
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Haversine公式
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c
    return distance


def find_nearest_point(
    target_lat: float,
    target_lon: float,
    points: List[Dict],
    lat_key: str = "lat",
    lon_key: str = "lon"
) -> Optional[Dict]:
    """
    ターゲット地点に最も近い点を検索
    
    Args:
        target_lat, target_lon: ターゲット地点の緯度・経度
        points: 候補点のリスト（各要素はlat/lonキーを持つdict）
        lat_key, lon_key: 緯度・経度のキー名
    
    Returns:
        最も近い点のdict（距離も含む）、見つからなければNone
    """
    if not points:
        return None
    
    min_distance = float('inf')
    nearest_point = None
    
    for point in points:
        if lat_key not in point or lon_key not in point:
            continue
        
        distance = haversine_distance(
            target_lat, target_lon,
            point[lat_key], point[lon_key]
        )
        
        if distance < min_distance:
            min_distance = distance
            nearest_point = point.copy()
            nearest_point['distance'] = min_distance
    
    return nearest_point


def normalize_hotel_name(name: str) -> str:
    """
    ホテル名を正規化（比較用）
    大文字小文字を統一、余分な空白を削除
    """
    if not name:
        return ""
    return " ".join(name.lower().split())


def safe_float(value, default: float = 0.0) -> float:
    """
    安全にfloatに変換
    """
    try:
        if isinstance(value, str):
            # 通貨記号やカンマを除去
            value = value.replace(",", "").replace("$", "").replace("€", "").replace("¥", "").strip()
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value, default: int = 0) -> int:
    """
    安全にintに変換
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

