import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LayerNormalization, Add, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine
import yaml
import os
import joblib

### 1. 데이터 수집 및 전처리
def get_db_connection():
    """SQLAlchemy 데이터베이스 엔진 생성"""
    with open('D:\\coding_project\\autotrade-main\\config.yaml', encoding='UTF-8') as f:
        _cfg = yaml.load(f, Loader=yaml.FullLoader)
    db_password = _cfg['DB_PASS'].replace("@", "%40").replace("#", "%23")
    db_url = f"mysql+pymysql://{_cfg['DB_USER']}:{db_password}@{_cfg['DB_HOST']}/{_cfg['DB_NAME']}"
    engine = create_engine(db_url)
    return engine

def fetch_symbols(engine):
    """stock_prices 테이블의 모든 고유한 심볼 가져오기"""
    query = "SELECT DISTINCT symbol FROM stock_prices"
    symbols = pd.read_sql(query, engine)['symbol'].tolist()
    return symbols

def fetch_all_data(engine, symbols):
    """모든 심볼에 대해 주가 및 지표 데이터를 가져와 하나의 데이터프레임으로 결합"""
    all_data = []
    for symbol in symbols:
        query = f"""
        SELECT sp.symbol, sp.date, sp.open, sp.high, sp.low, sp.close, sp.volume,
               ti.moving_average, ti.rsi, ti.macd, ti.macd_signal, ti.bollinger_upper,
               ti.bollinger_lower, ti.stochastic_k, ti.stochastic_d, ti.atr, ti.cci, ti.obv
        FROM stock_prices sp
        LEFT JOIN technical_indicators ti ON sp.symbol = ti.symbol AND sp.date = ti.date
        WHERE sp.symbol = '{symbol}'
        ORDER BY sp.date DESC
        LIMIT 20
        """
        df = pd.read_sql(query, engine)
        df['symbol'] = symbol
        all_data.append(df)

    combined_df = pd.concat(all_data)
    return combined_df

def preprocess_data(df):
    """데이터 전처리 및 스케일링"""
    combined_data = df[['open', 'high', 'low', 'close', 'volume', 'moving_average', 'rsi', 'macd', 
                        'macd_signal', 'bollinger_upper', 'bollinger_lower', 'stochastic_k', 
                        'stochastic_d', 'atr', 'cci', 'obv']].copy()
    combined_data.fillna(method='ffill', inplace=True)
    combined_data.fillna(method='bfill', inplace=True)
    combined_data.dropna(inplace=True)

    # 데이터 스케일링
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(combined_data)

    # 시계열 데이터 생성
    X, y = [], []
    sequence_length = 20  # 각 샘플의 길이를 20일치 데이터로 설정
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])  # 20일치 데이터를 입력으로 사용
        y.append(1 if scaled_data[i, 3] > scaled_data[i-1, 3] else 0)  # 상승/하락 라벨링

    X, y = np.array(X), np.array(y)
    return X, y, scaler

### Transformer 인코더 블록 정의
def simple_attention(inputs, head_size, dropout=0.2):
    """간단한 Attention 레이어"""
    query = Dense(head_size, activation="relu")(inputs)
    key = Dense(head_size, activation="relu")(inputs)
    value = Dense(head_size, activation="relu")(inputs)
    
    # Scaled dot-product attention
    scores = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.cast(head_size, tf.float32))
    attention_weights = tf.nn.softmax(scores, axis=-1)
    attention_output = tf.matmul(attention_weights, value)
    
    return Dropout(dropout)(attention_output)

def transformer_encoder(inputs, head_size, ff_dim, dropout=0.2):
    """간단한 Attention 기반 Transformer 인코더 블록"""
    attention = simple_attention(inputs, head_size, dropout)
    
    # Attention 출력의 shape을 LSTM 출력과 동일하게 맞추기 위해 Dense 레이어 추가
    attention = Dense(inputs.shape[-1])(attention)
    
    # Skip connection 추가
    attention = Add()([attention, inputs])  
    attention = LayerNormalization(epsilon=1e-6)(attention)

    outputs = Dense(ff_dim, activation="relu")(attention)
    outputs = Dense(inputs.shape[-1])(outputs)
    outputs = Add()([outputs, attention])  # Skip connection
    outputs = LayerNormalization(epsilon=1e-6)(outputs)
    
    return outputs


### 수정된 모델 구성 함수 (LSTM-CNN-Transformer 모델)
def build_lstm_cnn_transformer_model(sequence_length, features):
    """LSTM-CNN-Transformer 모델 생성"""
    inputs = Input(shape=(sequence_length, features))
    
    # Conv1D 레이어로 국소적 패턴 추출
    x = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    
    # LSTM 레이어로 시계열 특성 학습
    x = LSTM(units=100, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(units=100, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    
    # 간단한 Transformer 인코더 추가
    x = transformer_encoder(x, head_size=64, ff_dim=128, dropout=0.2)

    # Flatten 후, Dense 레이어로 연결하여 출력
    x = Flatten()(x)
    outputs = Dense(2, activation='softmax')(x)  # 상승과 하락 확률을 위한 Softmax 출력

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.00005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

### 모델 학습 및 평가
def train_and_evaluate(engine):
    """모든 종목 데이터를 학습하여 하나의 모델을 생성"""
    symbols = fetch_symbols(engine)
    combined_df = fetch_all_data(engine, symbols)
    X, y, scaler = preprocess_data(combined_df)

    # 수정된 모델 초기화 및 학습
    model = build_lstm_cnn_transformer_model(X.shape[1], X.shape[2])
    model.fit(X, y, epochs=20, batch_size=32, verbose=1, validation_split=0.2)
    
    # 모델 및 스케일러 저장
    os.makedirs("models", exist_ok=True)
    model.save("models/stock_LSTM_CNN_Transformer_model")  # 모델 저장
    joblib.dump(scaler, "models/scaler.joblib")  # 스케일러 저장
    print("모델과 스케일러가 저장되었습니다.")

# 실행 코드
engine = get_db_connection()
train_and_evaluate(engine)
