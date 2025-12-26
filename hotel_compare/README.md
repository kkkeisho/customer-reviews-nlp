# Hotel Compare 🏨

観光客目線でホテルを比較するStreamlitアプリケーション。

## 機能

- ✅ **ホテル検索**: Google Places APIで都市のホテルを検索
- ✅ **観光地取得**: Google Places APIで人気観光地を取得（評価×レビュー数でソート）
- ✅ **交通アクセス**: OSM Overpass APIで最寄駅・バス停を検索
- ✅ **OTAデータ統合**: CSVファイルでOTA（Booking, Expedia等）の価格・評価データを取り込み
- ✅ **Fuzzy Matching**: ホテル名の自動マッチング
- ✅ **レビュー分析**: 軽量モードまたはTransformersでレビュー要約・キーワード抽出
- ✅ **可視化**: OTA別の比較チャート、ヒートマップ

## セットアップ

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. APIキーの設定

`.env.example`を`.env`にコピーして、APIキーを設定してください。

```bash
cp .env.example .env
```

`.env`ファイルを編集：

```env
GOOGLE_PLACES_API_KEY=your_google_places_api_key_here
```

#### APIキーの取得方法

- **Google Places API** (必須): 
  1. [Google Cloud Console](https://console.cloud.google.com/)にアクセス
  2. プロジェクトを作成（または既存のプロジェクトを選択）
  3. 「APIとサービス」→「ライブラリ」を開く
  4. 「Places API」を検索して選択
  5. **「有効にする」ボタンをクリック**（重要！）
  6. 「APIとサービス」→「認証情報」を開く
  7. 「認証情報を作成」→「APIキー」を選択
  8. 作成されたAPIキーをコピーして`.env`ファイルに設定
  
  **重要**: APIキーを作成しただけでは不十分です。必ず「Places API」を有効化してください。
  
  **エラー「REQUEST_DENIED」が表示される場合**:
  - Places APIが有効化されているか確認
  - APIキーの制限設定を確認（「APIキーを制限」で「Places API」が許可されているか）

**注意**: 観光地の取得もGoogle Places APIを使用します。追加のAPIキーは不要です。

### 3. オプショナルパッケージ（NLP機能強化）

Transformersモードを使用する場合（レビュー要約・感情分析）：

```bash
pip install transformers torch
```

**注意**: Transformersはモデルダウンロードが必要で、初回実行時に時間がかかります。

## 起動方法

```bash
streamlit run app.py
```

ブラウザで `http://localhost:8501` が自動的に開きます。

## 使い方

### 基本的な使い方

1. **都市名を入力**: サイドバーで都市名（例: `Tokyo`, `Paris`, `New York`）を入力
2. **検索実行**: 「🔍 ホテルを検索」ボタンをクリック
3. **観光地を選択**: サイドバーで比較したい観光地を複数選択（デフォルトで上位3件）
4. **ホテルを選択**: テーブルからホテルをクリックして詳細を確認

### OTAデータの取り込み

1. **CSVファイルを準備**: 以下の形式でCSVファイルを作成

```csv
hotel_name,ota_name,rating,review_count,price,currency,review_text
Grand Hotel Tokyo,Booking,4.5,1234,15000,JPY,"Great location, clean rooms"
Grand Hotel Tokyo,Expedia,4.3,890,14500,JPY,"Nice hotel, friendly staff"
```

2. **CSVをアップロード**: サイドバーの「OTA CSVファイルをアップロード」からファイルを選択
3. **自動マッチング**: ホテル名がfuzzy matchingで自動的にマッチングされます

#### CSVフォーマット

| 列名 | 説明 | 必須 |
|------|------|------|
| `hotel_name` | OTA側のホテル名 | ✅ |
| `ota_name` | OTA名（例: Booking, Expedia） | ✅ |
| `rating` | 評価（0-5） | ⚠️ |
| `review_count` | レビュー数 | ⚠️ |
| `price` | 価格（数値または文字列） | ⚠️ |
| `currency` | 通貨コード | ⚠️ |
| `review_text` | レビューテキスト | ⚠️ |

**注意**: 列名は柔軟に対応します（`name`, `hotel`, `property_name`なども認識）

### レビュー分析モード

- **Lightweight**: 軽量キーワード抽出・簡易要約（デフォルト、高速）
- **Transformers (Optional)**: BART要約・感情分析（モデルダウンロード必要、高精度）

### フィルタ機能

サイドバーで以下を設定可能：
- 最小評価
- 最大価格
- 観光地までの最大距離

## プロジェクト構成

```
hotel_compare/
├── app.py                 # メインのStreamlitアプリ
├── config.py              # 設定管理
├── api_clients.py         # APIクライアント（Google Places, OSM）
├── data_processing.py     # データ処理（CSV読み込み、fuzzy matching）
├── nlp_utils.py           # NLPユーティリティ（レビュー要約）
├── utils.py               # ユーティリティ関数（距離計算など）
├── requirements.txt       # 依存パッケージ
├── .env.example           # 環境変数テンプレート
└── README.md             # このファイル
```

## 注意事項

### API制限

- **Google Places API**: リクエスト数に制限があります（無料枠あり、月間$200相当）
  - ホテル検索と観光地取得の両方に使用されます
- **OSM Overpass API**: 無料ですが、レート制限があります

### エラーハンドリング

- APIエラー時は適切なエラーメッセージを表示
- キャッシュ機能により、同じ検索条件ではAPI呼び出しをスキップ

### パフォーマンス

- Streamlitのキャッシュ機能（`@st.cache_data`）を使用してAPI呼び出しを最適化
- Transformersモードは初回実行時にモデルをダウンロード（数GB）

## トラブルシューティング

### ホテルが検索できない

- Google Places APIキーが正しく設定されているか確認
- 都市名のスペルを確認

### 観光地が取得できない

- Google Places APIキーが正しく設定されているか確認
- Places APIが有効化されているか確認（REQUEST_DENIEDエラーの場合）
- 都市名のスペルを確認
- 観光地が少ない都市の場合、結果が少なくなる可能性があります

### OTAデータがマッチングされない

- ホテル名の表記が大きく異なる場合は、fuzzy matchingの閾値を調整（`config.py`の`DEFAULT_FUZZY_MATCH_THRESHOLD`）
- CSVの列名を確認

### Transformersモードが動かない

- `pip install transformers torch` を実行
- 初回実行時はモデルダウンロードに時間がかかります
- メモリ不足の場合はLightweightモードを使用

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 貢献

バグ報告や機能要望はIssueでお知らせください。

