import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine
import yaml

def get_db_connection():
    """SQLAlchemy 데이터베이스 엔진 생성"""
    with open('D:\\coding_project\\autotrade-main\\config.yaml', encoding='UTF-8') as f:
        _cfg = yaml.load(f, Loader=yaml.FullLoader)
    db_password = _cfg['DB_PASS'].replace("@", "%40").replace("#", "%23")
    db_url = f"mysql+pymysql://{_cfg['DB_USER']}:{db_password}@{_cfg['DB_HOST']}/{_cfg['DB_NAME']}"
    engine = create_engine(db_url)
    return engine

def fetch_combined_data(engine, symbol):
    """20일치 주가 및 지표 데이터를 결합하여 가져오기"""
    query = f"""
    SELECT sp.date, sp.open, sp.high, sp.low, sp.close, sp.volume,
           ti.moving_average, ti.rsi, ti.macd, ti.macd_signal, ti.bollinger_upper,
           ti.bollinger_lower, ti.stochastic_k, ti.stochastic_d, ti.atr, ti.cci, ti.obv
    FROM stock_prices sp
    LEFT JOIN technical_indicators ti ON sp.symbol = ti.symbol AND sp.date = ti.date
    WHERE sp.symbol = '{symbol}'
    ORDER BY sp.date DESC
    LIMIT 20
    """
    combined_df = pd.read_sql(query, engine)
    return combined_df

def preprocess_combined_data(combined_df):
    """주가 및 지표 데이터 결합 후 스케일링 및 데이터셋 준비"""
    combined_data = combined_df[['open', 'high', 'low', 'close', 'volume', 'moving_average', 'rsi', 'macd', 
                                 'macd_signal', 'bollinger_upper', 'bollinger_lower', 'stochastic_k', 
                                 'stochastic_d', 'atr', 'cci', 'obv']].copy()

    # NaN 처리
    if combined_data.isnull().values.any():
        print("NaN values found. Replacing with median values.")
        combined_data.fillna(combined_data.median(), inplace=True)

    # 스케일링
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(combined_data)

    # 20일치 데이터로 입력과 출력 설정
    X = [scaled_data]
    y = scaled_data[-1, 3]  # 예측할 종가 (close)
    X, y = np.array(X), np.array([y])

    print(f"Processed data shapes - X: {X.shape}, y: {y.shape}")
    return X, y, scaler

def build_model(sequence_length, features):
    """LSTM 모델 생성"""
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(sequence_length, features)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
    return model

def train_on_all_symbols(engine):
    """모든 종목에 대해 학습"""
    symbols = pd.read_sql("SELECT DISTINCT symbol FROM stock_prices", engine)['symbol'].tolist()
    model = None
    for symbol in symbols:
        print(f"Training on symbol: {symbol}")
        
        combined_df = fetch_combined_data(engine, symbol)
        
        if len(combined_df) < 20:
            print(f"Not enough data for symbol: {symbol}")
            continue

        X, y, scaler = preprocess_combined_data(combined_df)
        
        if X.size == 0 or y.size == 0:
            print(f"No valid data for symbol: {symbol}")
            continue

        if model is None:
            model = build_model(X.shape[1], X.shape[2])

        model.fit(X, y, epochs=10, batch_size=1, verbose=1)

    model.save("stock_price_prediction_model")  # ".h5" 생략하여 SavedModel 형식으로 저장
    print("모델이 모든 종목에 대해 학습되었습니다.")

# 실행 코드
engine = get_db_connection()
train_on_all_symbols(engine)
