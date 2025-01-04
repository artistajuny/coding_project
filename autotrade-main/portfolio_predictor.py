# portfolio_predictor.py

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine
import joblib
import yaml

def get_db_connection():
    """SQLAlchemy 데이터베이스 엔진 생성"""
    with open('D:\\coding_project\\autotrade-main\\config.yaml', encoding='UTF-8') as f:
        _cfg = yaml.load(f, Loader=yaml.FullLoader)
    db_password = _cfg['DB_PASS'].replace("@", "%40").replace("#", "%23")
    db_url = f"mysql+pymysql://{_cfg['DB_USER']}:{db_password}@{_cfg['DB_HOST']}/{_cfg['DB_NAME']}"
    engine = create_engine(db_url)
    return engine

def fetch_recent_data(engine, symbol, days=30):
    """최근 days일치 주가 및 지표 데이터를 가져옵니다."""
    query = f"""
    SELECT sp.date, sp.open, sp.high, sp.low, sp.close, sp.volume,
           ti.moving_average, ti.rsi, ti.macd, ti.macd_signal, ti.bollinger_upper,
           ti.bollinger_lower, ti.stochastic_k, ti.stochastic_d, ti.atr, ti.cci, ti.obv
    FROM stock_prices sp
    LEFT JOIN technical_indicators ti ON sp.symbol = ti.symbol AND sp.date = ti.date
    WHERE sp.symbol = '{symbol}'
    ORDER BY sp.date DESC
    LIMIT {days}
    """
    df = pd.read_sql(query, engine)
    return df.iloc[::-1]  # 데이터 프레임을 최신 날짜 순으로 가져온 후 역순으로 변경

def preprocess_test_data(df, scaler):
    """테스트 데이터 전처리"""
    combined_data = df[['open', 'high', 'low', 'close', 'volume', 'moving_average', 'rsi', 'macd', 
                        'macd_signal', 'bollinger_upper', 'bollinger_lower', 'stochastic_k', 
                        'stochastic_d', 'atr', 'cci', 'obv']].copy()
    
    combined_data.fillna(method='ffill', inplace=True)
    combined_data.fillna(method='bfill', inplace=True)
    combined_data.dropna(inplace=True)

    scaled_data = scaler.transform(combined_data)  # 학습 때 사용한 스케일러로 변환
    X_test = np.array([scaled_data])  # 모델 입력 형태에 맞게 배열 생성
    return X_test

def load_model_and_scaler(model_type="LSTM_CNN_Transformer"):
    """모델과 스케일러 로드"""
    model_path = 'D:\\coding_project\\autotrade-main\\models_pri\\stock_LSTM_CNN_Transformer_model'
    scaler_path = 'D:\\coding_project\\autotrade-main\\models_pri\\scaler.joblib'

    
    try:
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)  # 학습 시 저장된 스케일러 로드
        print("모델과 스케일러가 성공적으로 로드되었습니다.")
        return model, scaler
    except Exception as e:
        print(f"모델 또는 스케일러 로드 중 오류 발생: {e}")
        return None, None


def predict_next_day(symbol, engine, model, scaler):
    test_data = fetch_recent_data(engine, symbol)
    
    if test_data is None or test_data.empty:
        return {"symbol": symbol, "rise_prob": None, "fall_prob": None}

    X_test = preprocess_test_data(test_data, scaler)
    
    prediction = model.predict(X_test)
    rise_prob, fall_prob = prediction[0]  # 상승과 하락 확률 추출

    return {"symbol": symbol, "rise_prob": rise_prob, "fall_prob": fall_prob}

def predict_all_portfolio(model_type="LSTM_CNN_Transformer"):
    engine = get_db_connection()
    model, scaler = load_model_and_scaler(model_type)
    
    if model is None or scaler is None:
        print("모델 또는 스케일러 로드에 실패하여 예측을 중단합니다.")
        return []

    latest_date_query = "SELECT MAX(date) FROM portfolio"
    latest_date = pd.read_sql(latest_date_query, engine).iloc[0, 0]
    
    if latest_date is None:
        print("포트폴리오 테이블에 데이터가 없습니다.")
        return []

    query = f"SELECT DISTINCT symbol FROM portfolio WHERE date = '{latest_date}'"
    symbols = pd.read_sql(query, engine)['symbol'].tolist()
    
    predictions = []
    for symbol in symbols:
        result = predict_next_day(symbol, engine, model, scaler)
        predictions.append(result)
    
    return predictions

if __name__ == "__main__":
    # 직접 실행되지 않도록 main block을 추가
    print("\nLSTM-CNN-Transformer 모델 예측 결과:")
    lstm_cnn_transformer_predictions = predict_all_portfolio("LSTM_CNN_Transformer")
    for prediction in lstm_cnn_transformer_predictions:
        print(prediction)