import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LayerNormalization, Add
import yaml
import os
import time
import joblib

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # TensorFlow를 CPU 모드에서 실행

# 데이터베이스 연결 설정
with open('D:\\coding_project\\autotrade-main\\config.yaml', encoding='UTF-8') as f:
    _cfg = yaml.load(f, Loader=yaml.FullLoader)

def get_db_connection():
    db_password = _cfg['DB_PASS'].replace("@", "%40").replace("#", "%23")
    db_url = f"mysql+pymysql://{_cfg['DB_USER']}:{db_password}@{_cfg['DB_HOST']}/{_cfg['DB_NAME']}"
    engine = create_engine(db_url)
    return engine

# 모든 심볼 데이터 결합 및 전처리
def fetch_all_data(engine):
    query = """
    SELECT sd.symbol, sd.timestamp, sd.price, 
           ti.rsi, ti.moving_average, ti.macd, ti.macd_signal, 
           ti.bollinger_upper, ti.bollinger_lower, sd.trade_strength, sd.trade_volume, sd.change_rate
    FROM stock_tick_data sd
    LEFT JOIN tick_indicators ti ON sd.symbol = ti.symbol AND sd.timestamp = ti.timestamp
    WHERE sd.is_active = 1
    ORDER BY sd.symbol, sd.timestamp DESC
    """
    df = pd.read_sql(query, engine)
    
    # 모든 값이 있는 행만 필터링
    df.dropna(inplace=True)
    df = df.drop(columns=['timestamp'])  # timestamp 열 제거
    
    # 필요한 열만 선택 (10개 열)
    numeric_columns = [
        'price', 'rsi', 'moving_average', 'macd', 'macd_signal', 
        'bollinger_upper', 'bollinger_lower', 'trade_strength', 'trade_volume', 'change_rate'
    ]
    df = df[numeric_columns]  # 초과하는 열 제거
    print("최종 데이터 열 확인:", df.columns.tolist())
    
    # 데이터 타입 설정
    df[numeric_columns] = df[numeric_columns].astype(np.float32)
    print("결합 데이터에 NaN 값이 있는지 확인:", df.isna().sum().sum())  # NaN이 남아 있는지 확인
    
    # 데이터 스케일링
    scaler = MinMaxScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    return df, scaler

# LSTM, CNN 및 대체 Attention 모델 설계
def build_lstm_cnn_attention_model(input_shape):
    inputs = Input(shape=input_shape)

    # LSTM Layer
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)

    # 대체 Attention Layer (간단한 Dense 레이어 사용)
    attention_weights = Dense(64, activation='tanh')(x)
    attention_output = Add()([x, attention_weights])  # Residual connection
    x = LayerNormalization(epsilon=1e-6)(attention_output)

    # CNN Layer
    x = Conv1D(32, kernel_size=3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    
    # Fully Connected Layers
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)  # 상승/하락 예측

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 배치 학습 함수
def train_model_batch(model, data, epochs=10, batch_size=16):
    state_size = data.shape[1]
    if state_size != 10:
        raise ValueError(f"데이터 특성 수가 10개가 아닙니다: {state_size}개")
    epoch_times = []

    for epoch in range(epochs):
        start_time = time.time()
        
        # 배치 데이터로 나누어 학습
        for i in range(0, len(data) - 20, batch_size):
            if i + batch_size + 20 > len(data):
                # 남은 데이터가 충분하지 않을 경우 배치를 건너뜀
                break
            batch_data = data.iloc[i:i + batch_size + 20]
            states = np.array([batch_data.iloc[j:j + 20].values for j in range(batch_size)])
            labels = np.array([
                1 if batch_data.iloc[j + 20]['price'] > batch_data.iloc[j + 19]['price'] else 0
                for j in range(batch_size)
            ])
            model.train_on_batch(states, labels)
        
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        print(f"Epoch {epoch + 1}/{epochs} completed in {epoch_time:.2f}s. Average time per epoch: {avg_epoch_time:.2f}s")

# 학습된 모델 확인 함수
def inspect_trained_model(model, scaler):
    print("\n--- 모델 구조 ---")
    model.summary()
    print("\n--- 모델 입력 형태 ---")
    print(f"Input shape: {model.input_shape}")
    print("\n--- 모델 컴파일 정보 ---")
    print(f"Loss function: {model.loss}")
    print(f"Optimizer: {model.optimizer}")
    print(f"Metrics: {model.metrics}")
    
    # 스케일러 정보
    print("\n--- MinMaxScaler 정보 ---")
    if hasattr(scaler, "feature_names_in_"):
        print(f"Scaler features: {scaler.feature_names_in_}")
    else:
        print("Scaler does not have feature names.")
    print(f"Scaler data min: {scaler.data_min_}")
    print(f"Scaler data max: {scaler.data_max_}")

# 실행 코드
if __name__ == "__main__":
    engine = get_db_connection()
    data, scaler = fetch_all_data(engine)
    if not data.empty:
        state_size = data.shape[1]  # 데이터 열 수 확인
        if state_size != 10:
            print(f"데이터 특성 수가 올바르지 않습니다. 현재 특성 수: {state_size}")
        else:
            model = build_lstm_cnn_attention_model((20, state_size))
            train_model_batch(model, data, epochs=10, batch_size=16)
            print("모든 심볼에 대한 통합 LSTM-CNN-Attention 모델 학습 완료")
            
            os.makedirs("models4", exist_ok=True)
            model.save("models4/unified_lstm_cnn_attention_model4")
            joblib.dump(scaler, "models4/scaler4.joblib")  # 스케일러 저장
            
            print("통합 LSTM-CNN-Attention 모델이 저장되었습니다.")
            
            # 모델 및 스케일러 확인
            inspect_trained_model(model, scaler)
