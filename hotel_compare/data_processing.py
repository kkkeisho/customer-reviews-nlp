"""
データ処理モジュール
CSV読み込み、fuzzy matching、データマージなど
"""
import pandas as pd
from typing import List, Dict, Optional
import utils
import config

try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    print("Warning: rapidfuzz not installed. Using simple string matching.")


def load_ota_csv(file) -> pd.DataFrame:
    """
    OTA CSVファイルを読み込む
    
    Args:
        file: Streamlitのuploaded_fileオブジェクト
    
    Returns:
        DataFrame
    """
    try:
        # エンコーディングを自動検出
        df = pd.read_csv(file, encoding='utf-8')
        
        # 列名の正規化（大文字小文字、空白を統一）
        df.columns = df.columns.str.strip().str.lower()
        
        # 必須列のマッピング（柔軟に対応）
        column_mapping = {
            'hotel_name': ['hotel_name', 'name', 'hotel', 'property_name'],
            'ota_name': ['ota_name', 'ota', 'platform', 'source'],
            'rating': ['rating', 'score', 'stars', 'review_score'],
            'review_count': ['review_count', 'reviews', 'num_reviews', 'total_reviews'],
            'price': ['price', 'rate', 'cost', 'amount'],
            'currency': ['currency', 'curr', 'ccy'],
            'review_text': ['review_text', 'review', 'reviews', 'comment', 'comments']
        }
        
        # 列名を標準化
        for standard_name, possible_names in column_mapping.items():
            for possible in possible_names:
                if possible in df.columns and standard_name not in df.columns:
                    df.rename(columns={possible: standard_name}, inplace=True)
                    break
        
        return df
        
    except Exception as e:
        raise ValueError(f"CSV読み込みエラー: {str(e)}")


def fuzzy_match_hotels(
    hotel_name: str,
    ota_hotel_names: List[str],
    threshold: int = None
) -> Optional[Dict]:
    """
    ホテル名をfuzzy matchingでマッチング
    
    Args:
        hotel_name: マッチング対象のホテル名
        ota_hotel_names: OTA側のホテル名リスト
        threshold: マッチング閾値（0-100、デフォルトはconfigから）
    
    Returns:
        {"matched_name": ..., "score": ...} または None
    """
    if threshold is None:
        threshold = config.DEFAULT_FUZZY_MATCH_THRESHOLD
    
    if not ota_hotel_names:
        return None
    
    normalized_target = utils.normalize_hotel_name(hotel_name)
    
    if RAPIDFUZZ_AVAILABLE:
        # rapidfuzzを使用
        result = process.extractOne(
            normalized_target,
            [utils.normalize_hotel_name(name) for name in ota_hotel_names],
            scorer=fuzz.ratio
        )
        
        if result and result[1] >= threshold:
            matched_idx = ota_hotel_names.index([n for n in ota_hotel_names if utils.normalize_hotel_name(n) == result[0]][0])
            return {
                "matched_name": ota_hotel_names[matched_idx],
                "score": result[1]
            }
    else:
        # 簡易版：部分一致と正規化比較
        best_score = 0
        best_match = None
        
        for ota_name in ota_hotel_names:
            normalized_ota = utils.normalize_hotel_name(ota_name)
            
            # 完全一致
            if normalized_target == normalized_ota:
                score = 100
            # 部分一致（一方が他方を含む）
            elif normalized_target in normalized_ota or normalized_ota in normalized_target:
                score = 80
            # 単語レベルでの一致
            elif set(normalized_target.split()) & set(normalized_ota.split()):
                score = 60
            else:
                score = 0
            
            if score > best_score:
                best_score = score
                best_match = ota_name
        
        if best_score >= threshold:
            return {
                "matched_name": best_match,
                "score": best_score
            }
    
    return None


def merge_ota_data(
    hotels: List[Dict],
    ota_df: pd.DataFrame
) -> List[Dict]:
    """
    ホテルリストにOTAデータをマージ
    
    Args:
        hotels: ホテル情報のリスト
        ota_df: OTAデータのDataFrame
    
    Returns:
        OTAデータがマージされたホテルリスト
    """
    if ota_df.empty:
        return hotels
    
    # 各ホテルに対してマッチング
    for hotel in hotels:
        hotel['ota_data'] = {}
        
        # OTAごとにグループ化（ota_name列がある場合）
        if 'ota_name' in ota_df.columns:
            ota_groups = ota_df.groupby('ota_name')
            for ota_name, group in ota_groups:
                ota_hotel_names = group['hotel_name'].tolist() if 'hotel_name' in group.columns else []
                
                match_result = fuzzy_match_hotels(hotel['name'], ota_hotel_names)
                
                if match_result:
                    # マッチしたOTAレコードを取得
                    matched_rows = group[group['hotel_name'] == match_result['matched_name']]
                    if not matched_rows.empty:
                        matched_row = matched_rows.iloc[0]
                        
                        ota_info = {
                            "rating": utils.safe_float(matched_row.get('rating', 0)),
                            "review_count": utils.safe_int(matched_row.get('review_count', 0)),
                            "price": matched_row.get('price', ''),
                            "currency": matched_row.get('currency', ''),
                            "review_text": matched_row.get('review_text', ''),
                            "match_score": match_result['score']
                        }
                        
                        hotel['ota_data'][ota_name] = ota_info
        else:
            # グループ化できない場合（ota_name列がない）
            ota_hotel_names = ota_df['hotel_name'].tolist() if 'hotel_name' in ota_df.columns else []
            match_result = fuzzy_match_hotels(hotel['name'], ota_hotel_names)
            
            if match_result:
                matched_rows = ota_df[ota_df['hotel_name'] == match_result['matched_name']]
                if not matched_rows.empty:
                    matched_row = matched_rows.iloc[0]
                    ota_name = matched_row.get('ota_name', 'Unknown')
                    
                    ota_info = {
                        "rating": utils.safe_float(matched_row.get('rating', 0)),
                        "review_count": utils.safe_int(matched_row.get('review_count', 0)),
                        "price": matched_row.get('price', ''),
                        "currency": matched_row.get('currency', ''),
                        "review_text": matched_row.get('review_text', ''),
                        "match_score": match_result['score']
                    }
                    
                    hotel['ota_data'][ota_name] = ota_info
    
    return hotels


def calculate_distances_to_attractions(
    hotels: List[Dict],
    selected_attractions: List[Dict]
) -> List[Dict]:
    """
    各ホテルから選択された観光地への距離を計算
    
    Args:
        hotels: ホテルリスト
        selected_attractions: 選択された観光地リスト
    
    Returns:
        距離情報が追加されたホテルリスト
    """
    for hotel in hotels:
        hotel_lat = hotel.get('lat', 0.0)
        hotel_lon = hotel.get('lon', 0.0)
        
        distances = []
        for attraction in selected_attractions:
            dist = utils.haversine_distance(
                hotel_lat, hotel_lon,
                attraction.get('lat', 0.0),
                attraction.get('lon', 0.0)
            )
            distances.append({
                "name": attraction.get('name', ''),
                "distance": dist
            })
        
        # 最小距離と平均距離を計算
        if distances:
            min_dist = min(d['distance'] for d in distances)
            avg_dist = sum(d['distance'] for d in distances) / len(distances)
            hotel['min_attraction_distance'] = min_dist
            hotel['avg_attraction_distance'] = avg_dist
            hotel['attraction_distances'] = distances
        else:
            hotel['min_attraction_distance'] = None
            hotel['avg_attraction_distance'] = None
            hotel['attraction_distances'] = []
    
    return hotels


def add_transport_info(
    hotels: List[Dict],
    transport_data: Dict[str, List[Dict]]
) -> List[Dict]:
    """
    各ホテルに最寄駅・バス停情報を追加
    
    Args:
        hotels: ホテルリスト
        transport_data: {"stations": [...], "bus_stops": [...]}
    
    Returns:
        交通情報が追加されたホテルリスト
    """
    stations = transport_data.get('stations', [])
    bus_stops = transport_data.get('bus_stops', [])
    
    for hotel in hotels:
        hotel_lat = hotel.get('lat', 0.0)
        hotel_lon = hotel.get('lon', 0.0)
        
        # 最寄駅
        nearest_station = utils.find_nearest_point(hotel_lat, hotel_lon, stations)
        if nearest_station:
            hotel['nearest_station'] = nearest_station.get('name', 'Unknown')
            hotel['nearest_station_distance'] = nearest_station.get('distance', 0.0)
        else:
            hotel['nearest_station'] = None
            hotel['nearest_station_distance'] = None
        
        # 最寄バス停
        nearest_bus_stop = utils.find_nearest_point(hotel_lat, hotel_lon, bus_stops)
        if nearest_bus_stop:
            hotel['nearest_bus_stop'] = nearest_bus_stop.get('name', 'Unknown')
            hotel['nearest_bus_stop_distance'] = nearest_bus_stop.get('distance', 0.0)
        else:
            hotel['nearest_bus_stop'] = None
            hotel['nearest_bus_stop_distance'] = None
    
    return hotels

