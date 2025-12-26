"""
Hotel Compare - Streamlitアプリ
観光客目線でホテルを比較するアプリケーション
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional

import config
import api_clients
import data_processing
import nlp_utils
import utils

# ページ設定
st.set_page_config(
    page_title="Hotel Compare",
    page_icon="🏨",
    layout="wide"
)

# タイトル
st.title("🏨 Hotel Compare")
st.markdown("観光客目線でホテルを比較するツール")


# ==================== サイドバー ====================
st.sidebar.header("設定")

# 都市名入力
city = st.sidebar.text_input("都市名", value="Tokyo", help="例: Tokyo, Paris, New York")

# 検索件数
col1, col2 = st.sidebar.columns(2)
hotel_count = col1.number_input("ホテル数", min_value=10, max_value=100, value=30, step=5)
attraction_count = col2.number_input("観光地数", min_value=5, max_value=50, value=15, step=5)

# 検索実行ボタン
search_button = st.sidebar.button("🔍 ホテルを検索", type="primary")

# OTA CSVアップロード
st.sidebar.markdown("---")
st.sidebar.subheader("OTAデータ")
ota_file = st.sidebar.file_uploader(
    "OTA CSVファイルをアップロード",
    type=["csv"],
    help="CSV形式: hotel_name, ota_name, rating, review_count, price, currency, review_text"
)

# NLPモード選択
st.sidebar.markdown("---")
st.sidebar.subheader("レビュー分析")
nlp_mode = st.sidebar.radio(
    "NLPモード",
    ["Lightweight", "Transformers (Optional)"],
    help="Lightweight: 軽量キーワード抽出 / Transformers: 要約・感情分析（モデルダウンロード必要）"
)

# フィルタ
st.sidebar.markdown("---")
st.sidebar.subheader("フィルタ")
filter_rating_min = st.sidebar.slider("最小評価", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
filter_price_max = st.sidebar.number_input("最大価格", min_value=0, value=0, help="0は無制限")
filter_distance_max = st.sidebar.number_input("最大距離（km）", min_value=0.0, value=0.0, step=0.5, help="観光地までの最大距離（0は無制限）")


# ==================== データ取得（キャッシュ付き） ====================
@st.cache_data
def fetch_hotels_cached(city: str, hotel_count: int):
    """ホテルデータを取得（キャッシュ）"""
    try:
        hotels = api_clients.fetch_google_places_hotels(city, hotel_count)
        return hotels, None
    except Exception as e:
        return [], str(e)


@st.cache_data
def fetch_attractions_cached(city: str, attraction_count: int):
    """観光地データを取得（キャッシュ）"""
    try:
        attractions = api_clients.fetch_attractions_google(
            city_name=city,
            radius_m=10000,  # 10km
            max_results=attraction_count
        )
        return attractions, None
    except Exception as e:
        return [], str(e)


@st.cache_data
def fetch_transport_cached(hotels: List[Dict], radius_km: float):
    """交通情報を取得（キャッシュ）"""
    transport_data_all = {}
    
    for hotel in hotels:
        hotel_id = hotel.get('place_id', '')
        if hotel_id not in transport_data_all:
            transport_data = api_clients.fetch_osm_stations_and_stops(
                hotel.get('lat', 0.0),
                hotel.get('lon', 0.0),
                radius_km
            )
            transport_data_all[hotel_id] = transport_data
    
    return transport_data_all


# ==================== メイン処理 ====================
if search_button or 'hotels' in st.session_state:
    # ホテル検索
    if search_button:
        with st.spinner("ホテルを検索中..."):
            hotels, error = fetch_hotels_cached(city, hotel_count)
            if error:
                st.error(f"エラー: {error}")
                st.stop()
            st.session_state['hotels'] = hotels
            st.session_state['city'] = city
    
    hotels = st.session_state.get('hotels', [])
    
    if not hotels:
        st.warning("ホテルが見つかりませんでした。")
        st.stop()
    
    # 観光地取得
    with st.spinner("観光地を取得中..."):
        attractions, error = fetch_attractions_cached(city, attraction_count)
        if error:
            st.warning(f"観光地取得エラー: {error}")
            attractions = []
    
    # 観光地選択UI
    st.sidebar.markdown("---")
    st.sidebar.subheader("観光地選択")
    
    if attractions:
        # 評価×レビュー数でソート済みなので、上位3件をデフォルト選択
        # ただし、ratingとuser_ratings_totalが高い順に並んでいることを前提
        attraction_names = [
            f"{attr.get('name', 'Unknown')} ({attr.get('kinds', 'tourist_attraction')[:30]}...)" 
            for attr in attractions
        ]
        selected_indices = st.sidebar.multiselect(
            "比較対象の観光地を選択",
            options=range(len(attractions)),
            format_func=lambda x: attraction_names[x],
            default=list(range(min(3, len(attractions))))  # デフォルトで上位3件
        )
        selected_attractions = [attractions[i] for i in selected_indices]
    else:
        selected_attractions = []
        st.sidebar.info("観光地データがありません")
    
    # 交通情報取得
    with st.spinner("交通情報を取得中..."):
        transport_data_all = fetch_transport_cached(hotels, config.DEFAULT_SEARCH_RADIUS_KM)
    
    # 距離計算
    hotels = data_processing.calculate_distances_to_attractions(hotels, selected_attractions)
    
    # 交通情報を追加
    for hotel in hotels:
        hotel_id = hotel.get('place_id', '')
        transport_data = transport_data_all.get(hotel_id, {"stations": [], "bus_stops": []})
        data_processing.add_transport_info([hotel], transport_data)
    
    # OTAデータマージ
    if ota_file is not None:
        try:
            ota_df = data_processing.load_ota_csv(ota_file)
            hotels = data_processing.merge_ota_data(hotels, ota_df)
            st.sidebar.success(f"OTAデータを読み込みました（{len(ota_df)}件）")
        except Exception as e:
            st.sidebar.error(f"OTAデータ読み込みエラー: {str(e)}")
    
    # フィルタ適用
    filtered_hotels = hotels.copy()
    
    if filter_rating_min > 0:
        filtered_hotels = [h for h in filtered_hotels if h.get('rating', 0) >= filter_rating_min]
    
    if filter_distance_max > 0:
        filtered_hotels = [
            h for h in filtered_hotels
            if h.get('min_attraction_distance') is not None
            and h.get('min_attraction_distance', float('inf')) <= filter_distance_max
        ]
    
    # ==================== メイン表示 ====================
    st.header(f"🏨 {city} のホテル一覧")
    st.caption(f"{len(filtered_hotels)}件のホテルが見つかりました")
    
    # テーブル用データ準備
    table_data = []
    for hotel in filtered_hotels:
        row = {
            "ホテル名": hotel.get('name', ''),
            "住所": hotel.get('address', '')[:50] + "..." if len(hotel.get('address', '')) > 50 else hotel.get('address', ''),
            "Google評価": f"{hotel.get('rating', 0):.1f} ⭐",
            "レビュー数": hotel.get('user_ratings_total', 0),
        }
        
        # 観光地距離
        if hotel.get('min_attraction_distance') is not None:
            row["観光地距離(km)"] = f"{hotel.get('min_attraction_distance', 0):.2f}"
        else:
            row["観光地距離(km)"] = "N/A"
        
        # 最寄駅
        if hotel.get('nearest_station_distance') is not None:
            row["最寄駅"] = f"{hotel.get('nearest_station', 'Unknown')} ({hotel.get('nearest_station_distance', 0):.2f}km)"
        else:
            row["最寄駅"] = "N/A"
        
        # 最寄バス停
        if hotel.get('nearest_bus_stop_distance') is not None:
            row["最寄バス停"] = f"{hotel.get('nearest_bus_stop', 'Unknown')} ({hotel.get('nearest_bus_stop_distance', 0):.2f}km)"
        else:
            row["最寄バス停"] = "N/A"
        
        # OTAデータ
        ota_data = hotel.get('ota_data', {})
        if ota_data:
            ota_info_list = []
            for ota_name, ota_info in ota_data.items():
                rating = ota_info.get('rating', 0)
                price = ota_info.get('price', '')
                ota_info_list.append(f"{ota_name}: {rating:.1f}⭐ / {price}")
            row["OTA情報"] = " | ".join(ota_info_list)
        else:
            row["OTA情報"] = "-"
        
        table_data.append(row)
    
    # テーブル表示
    df = pd.DataFrame(table_data)
    
    # ホテル選択用のセレクトボックス
    hotel_names = [f"{i+1}. {hotel.get('name', 'Unknown')}" for i, hotel in enumerate(filtered_hotels)]
    if 'selected_hotel_idx' not in st.session_state:
        st.session_state['selected_hotel_idx'] = 0
    
    selected_hotel_name = st.selectbox(
        "詳細を表示するホテルを選択",
        options=range(len(hotel_names)),
        format_func=lambda x: hotel_names[x],
        index=st.session_state.get('selected_hotel_idx', 0),
        key="hotel_selector"
    )
    
    # テーブル表示
    st.dataframe(df, use_container_width=True)
    
    # 選択されたホテルの詳細表示
    if selected_hotel_name is not None:
        selected_idx = selected_hotel_name
        selected_hotel = filtered_hotels[selected_idx]
        st.session_state['selected_hotel_idx'] = selected_idx
        
        st.markdown("---")
        st.header(f"📋 {selected_hotel.get('name', '')} の詳細")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("基本情報")
            st.write(f"**住所**: {selected_hotel.get('address', 'N/A')}")
            st.write(f"**Google評価**: {selected_hotel.get('rating', 0):.1f} ⭐ ({selected_hotel.get('user_ratings_total', 0)}件のレビュー)")
            st.write(f"**座標**: ({selected_hotel.get('lat', 0):.4f}, {selected_hotel.get('lon', 0):.4f})")
            
            if selected_hotel.get('min_attraction_distance') is not None:
                st.write(f"**観光地までの最短距離**: {selected_hotel.get('min_attraction_distance', 0):.2f} km")
                st.write(f"**観光地までの平均距離**: {selected_hotel.get('avg_attraction_distance', 0):.2f} km")
            
            if selected_hotel.get('nearest_station'):
                st.write(f"**最寄駅**: {selected_hotel.get('nearest_station', 'N/A')} ({selected_hotel.get('nearest_station_distance', 0):.2f} km)")
            
            if selected_hotel.get('nearest_bus_stop'):
                st.write(f"**最寄バス停**: {selected_hotel.get('nearest_bus_stop', 'N/A')} ({selected_hotel.get('nearest_bus_stop_distance', 0):.2f} km)")
        
        with col2:
            st.subheader("地図")
            # 地図データ準備
            map_data = []
            
            # ホテル
            map_data.append({
                "lat": selected_hotel.get('lat', 0),
                "lon": selected_hotel.get('lon', 0),
                "name": selected_hotel.get('name', 'Hotel')
            })
            
            # 選択された観光地
            for attr in selected_attractions:
                map_data.append({
                    "lat": attr.get('lat', 0),
                    "lon": attr.get('lon', 0),
                    "name": attr.get('name', 'Attraction')
                })
            
            if map_data:
                map_df = pd.DataFrame(map_data)
                st.map(map_df, use_container_width=True)
        
        # OTAデータ詳細
        ota_data = selected_hotel.get('ota_data', {})
        if ota_data:
            st.subheader("OTA情報")
            
            for ota_name, ota_info in ota_data.items():
                with st.expander(f"📊 {ota_name}"):
                    st.write(f"**評価**: {ota_info.get('rating', 0):.1f} ⭐")
                    st.write(f"**レビュー数**: {ota_info.get('review_count', 0)}")
                    st.write(f"**価格**: {ota_info.get('price', 'N/A')} {ota_info.get('currency', '')}")
                    st.write(f"**マッチングスコア**: {ota_info.get('match_score', 0):.1f}%")
                    
                    # レビュー処理
                    review_text = ota_info.get('review_text', '')
                    if review_text:
                        st.write("**レビュー分析**")
                        review_result = nlp_utils.process_reviews(
                            str(review_text),
                            mode="transformers" if nlp_mode == "Transformers (Optional)" else "lightweight"
                        )
                        
                        st.write(f"**要約**: {review_result.get('summary', 'N/A')}")
                        st.write(f"**キーワード**: {', '.join(review_result.get('keywords', [])[:10])}")
                        st.write(f"**感情**: {review_result.get('sentiment', 'neutral')}")
                        
                        if review_result.get('sentiment_score', 0) > 0:
                            st.write(f"**感情スコア**: {review_result.get('sentiment_score', 0):.3f}")
        
        # 観光地距離の詳細
        if selected_hotel.get('attraction_distances'):
            st.subheader("観光地までの距離")
            dist_df = pd.DataFrame(selected_hotel.get('attraction_distances', []))
            if not dist_df.empty:
                st.dataframe(dist_df, use_container_width=True)
    
    # ==================== OTA比較可視化 ====================
    if ota_file is not None:
        st.markdown("---")
        st.header("📊 OTA比較分析")
        
        # OTA別の集計
        ota_stats = {}
        for hotel in filtered_hotels:
            ota_data = hotel.get('ota_data', {})
            for ota_name, ota_info in ota_data.items():
                if ota_name not in ota_stats:
                    ota_stats[ota_name] = {
                        "ratings": [],
                        "prices": [],
                        "review_counts": []
                    }
                
                rating = ota_info.get('rating', 0)
                if rating > 0:
                    ota_stats[ota_name]["ratings"].append(rating)
                
                price = utils.safe_float(ota_info.get('price', 0))
                if price > 0:
                    ota_stats[ota_name]["prices"].append(price)
                
                review_count = ota_info.get('review_count', 0)
                if review_count > 0:
                    ota_stats[ota_name]["review_counts"].append(review_count)
        
        if ota_stats:
            col1, col2 = st.columns(2)
            
            with col1:
                # 平均評価の比較
                avg_ratings = {
                    ota: sum(stats["ratings"]) / len(stats["ratings"])
                    for ota, stats in ota_stats.items()
                    if stats["ratings"]
                }
                
                if avg_ratings:
                    fig_rating = px.bar(
                        x=list(avg_ratings.keys()),
                        y=list(avg_ratings.values()),
                        labels={"x": "OTA", "y": "平均評価"},
                        title="OTA別平均評価"
                    )
                    st.plotly_chart(fig_rating, use_container_width=True)
            
            with col2:
                # 平均価格の比較
                avg_prices = {
                    ota: sum(stats["prices"]) / len(stats["prices"])
                    for ota, stats in ota_stats.items()
                    if stats["prices"]
                }
                
                if avg_prices:
                    fig_price = px.bar(
                        x=list(avg_prices.keys()),
                        y=list(avg_prices.values()),
                        labels={"x": "OTA", "y": "平均価格"},
                        title="OTA別平均価格"
                    )
                    st.plotly_chart(fig_price, use_container_width=True)
            
            # ヒートマップ（OTA × ホテル）
            st.subheader("OTA × ホテル 評価ヒートマップ")
            heatmap_data = []
            for hotel in filtered_hotels:
                hotel_name = hotel.get('name', '')
                ota_data = hotel.get('ota_data', {})
                for ota_name, ota_info in ota_data.items():
                    rating = ota_info.get('rating', 0)
                    if rating > 0:
                        heatmap_data.append({
                            "ホテル": hotel_name[:30],  # 長い名前を切り詰め
                            "OTA": ota_name,
                            "評価": rating
                        })
            
            if heatmap_data:
                heatmap_df = pd.DataFrame(heatmap_data)
                pivot_df = heatmap_df.pivot_table(
                    index="ホテル",
                    columns="OTA",
                    values="評価",
                    aggfunc="mean"
                )
                
                fig_heatmap = px.imshow(
                    pivot_df,
                    labels=dict(x="OTA", y="ホテル", color="評価"),
                    title="OTA × ホテル 評価ヒートマップ",
                    color_continuous_scale="RdYlGn"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)

else:
    # 初期状態
    st.info("👈 サイドバーから都市名を入力して「ホテルを検索」をクリックしてください")
    
    st.markdown("""
    ### 使い方
    
    1. **都市名を入力**: サイドバーで都市名（例: Tokyo, Paris）を入力
    2. **検索実行**: 「🔍 ホテルを検索」ボタンをクリック
    3. **観光地を選択**: サイドバーで比較したい観光地を選択（デフォルトで上位3件）
    4. **OTAデータをアップロード**: CSVファイルをアップロードすると、ホテルと自動マッチング
    5. **ホテルを選択**: テーブルからホテルをクリックして詳細を確認
    
    ### 機能
    
    - ✅ Google Places APIでホテル検索
    - ✅ Google Places APIで観光地取得
    - ✅ OSM Overpass APIで最寄駅・バス停検索
    - ✅ OTAデータのCSV取り込みとfuzzy matching
    - ✅ レビュー要約・キーワード抽出（軽量/Transformers）
    - ✅ OTA別の比較可視化
    """)

