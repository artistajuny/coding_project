import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LayerNormalization, Add, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
import yaml
import os
import joblib
import time

### 데이터베이스 연결
def get_db_connection():
    """SQLAlchemy 데이터베이스 엔진 생성"""
    print("Establishing database connection...")
    with open('D:\\coding_project\\autotrade-main\\config.yaml', encoding='UTF-8') as f:
        _cfg = yaml.safe_load(f)
    db_password = _cfg['DB_PASS'].replace("@", "%40").replace("#", "%23")
    db_url = f"mysql+pymysql://{_cfg['DB_USER']}:{db_password}@{_cfg['DB_HOST']}/{_cfg['DB_NAME']}"
    engine = create_engine(db_url)
    print("Database connection established.")
    return engine

def fetch_all_data(engine):
    """한 번의 쿼리로 모든 데이터를 가져오기"""
    print("Fetching data from the database...")
    start_time = time.time()
    query = """
    SELECT sp.symbol, sp.date, sp.open, sp.high, sp.low, sp.close, sp.volume,
           ti.moving_average, ti.rsi, ti.macd, ti.macd_signal, ti.bollinger_upper,
           ti.bollinger_lower, ti.stochastic_k, ti.stochastic_d, ti.atr, ti.cci, ti.obv,
           cd.day_of_week, cd.month, cd.quarter, cd.seasonal,
           ef.kospi, ef.kosdaq, ef.s_and_p_500, ef.oil_price, ef.gold_price, ef.vix,
           ef.usd_krw, ef.eur_usd, ef.nasdaq, lb.classification
    FROM ai_tr_stock_price sp
    LEFT JOIN technical_indicators ti ON sp.symbol = ti.symbol AND sp.date = ti.date
    LEFT JOIN cycle_data cd ON sp.symbol = cd.symbol AND sp.date = cd.date
    LEFT JOIN external_factors ef ON sp.date = ef.date
    LEFT JOIN labels lb ON sp.symbol = lb.symbol AND sp.date = lb.date
    ORDER BY sp.date DESC
    """
    combined_df = pd.read_sql(query, engine)
    print(f"Data fetched: {len(combined_df)} rows in {time.time() - start_time:.2f} seconds.")
  
    return combined_df

### 데이터 전처리
def preprocess_data(df, sequence_length=20):
    """데이터 전처리 및 스케일링"""
    print("Preprocessing data...")
    combined_data = df[['open', 'high', 'low', 'close', 'volume', 'moving_average', 'rsi', 'macd', 
                        'macd_signal', 'bollinger_upper', 'bollinger_lower', 'stochastic_k', 
                        'stochastic_d', 'atr', 'cci', 'obv', 'day_of_week', 'month', 'quarter', 
                        'seasonal', 'kospi', 'kosdaq', 's_and_p_500', 'oil_price', 
                        'gold_price', 'vix', 'usd_krw', 'eur_usd', 'nasdaq']].copy()

    # NaN 처리
    combined_data.fillna(method='ffill', inplace=True)
    combined_data.fillna(method='bfill', inplace=True)
    combined_data.dropna(inplace=True)

    # 데이터 스케일링
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(combined_data)

    # 시계열 데이터 생성 (벡터화 적용)
    num_samples = len(scaled_data) - sequence_length
    X = np.array([scaled_data[i:i+sequence_length] for i in range(num_samples)])
    y = df['classification'].values[sequence_length:]

    print(f"Preprocessed data shapes - X: {X.shape}, y: {y.shape}")
    return X, y, scaler

### 간단한 Attention 메커니즘
def simple_attention(inputs):
    """간단한 Attention 메커니즘"""
    query = Dense(inputs.shape[-1], activation="relu")(inputs)
    key = Dense(inputs.shape[-1], activation="relu")(inputs)
    value = Dense(inputs.shape[-1], activation="relu")(inputs)

    attention_scores = tf.matmul(query, key, transpose_b=True)
    attention_scores = tf.nn.softmax(attention_scores, axis=-1)
    attention_output = tf.matmul(attention_scores, value)

    return attention_output

### Transformer 인코더 블록 정의
def transformer_encoder(inputs, head_size, ff_dim, dropout=0.2):
    """Transformer 인코더 블록"""
    attention = simple_attention(inputs)
    attention = Dense(inputs.shape[-1])(attention)
    attention = Add()([attention, inputs])  # Skip connection
    attention = LayerNormalization(epsilon=1e-6)(attention)

    outputs = Dense(ff_dim, activation="relu")(attention)
    outputs = Dense(inputs.shape[-1])(outputs)
    outputs = Add()([outputs, attention])  # Skip connection
    outputs = LayerNormalization(epsilon=1e-6)(outputs)

    return outputs

### LSTM-CNN-Transformer 모델 생성
def build_lstm_cnn_transformer_model(sequence_length, features):
    """LSTM-CNN-Transformer 모델 생성"""
    print("Building the LSTM-CNN-Transformer model...")
    inputs = Input(shape=(sequence_length, features))
    x = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)
    x = LSTM(units=100, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = transformer_encoder(x, head_size=64, ff_dim=128, dropout=0.2)
    x = Flatten()(x)
    outputs = Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.00005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Model built successfully.")
    return model

### 모델 학습
def train_and_evaluate(engine):
    """모든 데이터를 학습하여 하나의 모델을 생성"""
    print("Starting the training process...")
    combined_df = fetch_all_data(engine)
    if combined_df.empty:
        print("No data available for training.")
        return

    X, y, scaler = preprocess_data(combined_df)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_lstm_cnn_transformer_model(X.shape[1], X.shape[2])
    print("Training the model...")
    model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_val, y_val), verbose=1)

    print("Saving the model and scaler...")
    os.makedirs("models250103", exist_ok=True)
    model.save("models250103/stock_LSTM_CNN_Transformer_model")
    joblib.dump(scaler, "models250103/scaler.joblib")
    print("Model and scaler saved successfully.")

# 실행 코드
engine = get_db_connection()
train_and_evaluate(engine)
