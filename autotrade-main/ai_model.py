import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# 학습된 LSTM-CNN-Attention 모델 로드
model = load_model('D:\\coding_project\\autotrade-main\\models4\\unified_lstm_cnn_attention_model4')
scaler = MinMaxScaler()

# 학습 데이터로 fit()한 scaler를 저장했다고 가정하고 이를 load
scaler = joblib.load('D:\\coding_project\\autotrade-main\\models4\\scaler4.joblib')

# RSI 계산 함수
def calculate_rsi(data, period=14):
    if len(data) < period:
        return 0.0  # 기본값 설정

    delta = np.diff(data)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    avg_gain = np.convolve(gain, np.ones((period,)) / period, mode='valid')
    avg_loss = np.convolve(loss, np.ones((period,)) / period, mode='valid')

    # avg_loss가 0일 때 기본값 설정
    if np.any(avg_loss == 0):
        return 50.0  # 변동이 없는 경우 RSI를 중립값으로 설정

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi[-1] if len(rsi) > 0 else None

# 이동평균 계산 함수
def calculate_moving_average(data, period=20):
    if len(data) < period:
        return 0.0  # 기본값 설정

    return np.convolve(data, np.ones((period,)) / period, mode='valid')[-1]
# EMA 계산 함수 (MACD 계산에 사용)
# MACD 계산 디버깅
def calculate_exponential_moving_average(data, period):
    ema = np.zeros_like(data)
    ema[0] = data[0]
    alpha = 2 / (period + 1)
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
    # print(f"EMA({period}) 계산: {ema[-5:]}")  # 마지막 5개 값 확인
    return ema

# MACD 계산 함수
def calculate_macd(data, short_period=12, long_period=26, signal_period=9):
    if len(data) < max(short_period, long_period):
        print(f"MACD 계산 실패: 데이터 길이 부족 (필요: {max(short_period, long_period)}, 제공: {len(data)})")
        return None, None
    short_ema = calculate_exponential_moving_average(data, short_period)
    long_ema = calculate_exponential_moving_average(data, long_period)
    macd = short_ema - long_ema
    signal = calculate_exponential_moving_average(macd, signal_period)[-1]  # 전체 MACD에 대해 Signal 계산
    # print(f"MACD 계산 성공: MACD={macd[-1]}, Signal={signal}")
    return macd[-1], signal


# 볼린저 밴드 계산 함수
def calculate_bollinger_bands(data, period=20, num_std_dev=2):
    if len(data) < period:
        return None, None
    rolling_mean = np.convolve(data, np.ones((period,)) / period, mode='valid')[-1]
    rolling_std = np.std(data[-period:])
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band

# 모델 예측 결과를 텍스트로 변환하는 함수
def classify_prediction(symbol, predicted_profit):
    if predicted_profit > 0.8:  # 80% 이상의 확률
        return {symbol: "강력 매수"}
    elif 0.5 <= predicted_profit <= 0.8:  # 60% ~ 80% 사이의 확률
        return {symbol: "매수"}
    elif 0.4 <= predicted_profit < 0.5:  # 40% ~ 60% 사이의 확률
        return {symbol: "중립"}
    elif 0.2 <= predicted_profit < 0.4:  # 20% ~ 40% 사이의 확률
        return {symbol: "매도"}
    else:  # 0% ~ 20% 사이의 확률
        return {symbol: "강력 매도"}
    
def analyze_signals_with_trade_data(symbol, trade_data):
    # 받은 데이터 출력
    # print(f"Received data for {symbol}:")
    # for i, entry in enumerate(trade_data):
    #     print(f"  [{i}] {entry}")

    # 필요한 데이터 추출
    prices = np.array([entry['price'] for entry in trade_data if entry['price'] is not None])
    trade_volume = np.array([entry['trade_volume'] for entry in trade_data if entry['trade_volume'] is not None])
    trade_strengths = np.array([entry['trade_strength'] for entry in trade_data if entry['trade_strength'] is not None])
    change_rate = np.array([entry['change_rate'] for entry in trade_data if entry['change_rate'] is not None])

    # 데이터 길이 확인
    # print(f"{symbol} - Data Lengths: Prices={len(prices)}, Volume={len(trade_volume)}, Strengths={len(trade_strengths)}, Change Rate={len(change_rate)}")

    # 지표 계산
    try:
        rsi = calculate_rsi(prices)
        moving_average = calculate_moving_average(prices, 20)
        macd, macd_signal = calculate_macd(prices)
        bollinger_upper, bollinger_lower = calculate_bollinger_bands(prices)

        # print(f"{symbol} - Calculated Indicators:")
        # print(f"  RSI: {rsi}")
        # print(f"  Moving Average: {moving_average}")
        # print(f"  MACD: {macd}, Signal: {macd_signal}")
        # print(f"  Bollinger Bands: Upper={bollinger_upper}, Lower={bollinger_lower}")

    except Exception as e:
        print(f"{symbol} - Indicator Calculation Error: {e}")
        return {symbol: "지표 계산 실패"}

    # 특성 배열 생성
    if None not in [rsi, moving_average, macd, macd_signal, bollinger_upper, bollinger_lower]:
        try:
            # 특성 배열 생성
            features = np.array([
                prices[-20:],  # 가격
                np.full(20, rsi),  # RSI
                np.full(20, moving_average),  # 이동평균
                np.full(20, macd),  # MACD
                np.full(20, macd_signal),  # MACD Signal
                np.full(20, bollinger_upper),  # 볼린저 상한
                np.full(20, bollinger_lower),  # 볼린저 하한
                trade_strengths[-20:],  # 거래 강도
                trade_volume[-20:],  # 거래량
                change_rate[-20:]  # 변동률
            ]).T  # 전치행렬로 특성을 2D 배열로 변환

            # print(f"{symbol} - Features Shape Before Scaling: {features.shape}")

            # DataFrame 변환
            features_df = pd.DataFrame(features, columns=[
                'price', 'rsi', 'moving_average', 'macd', 'macd_signal', 
                'bollinger_upper', 'bollinger_lower', 'trade_strength', 'trade_volume', 'change_rate'
            ])

            # MinMaxScaler로 스케일링
            features_scaled = scaler.transform(features_df)
            # print(f"{symbol} - Scaled Features Shape: {features_scaled.shape}")

            # 모델 입력 형식으로 변환
            features_scaled = features_scaled.reshape(1, 20, -1)
            # print(f"{symbol} - Reshaped Features for Model: {features_scaled.shape}")

            # 모델 예측
            predicted_profit = model.predict(features_scaled)[0][0]
            # print(f"{symbol} - Predicted Profit: {predicted_profit}")

            # 예측 결과를 텍스트로 변환
            classification = classify_prediction(symbol, predicted_profit)
            print(f"Symbol: {symbol}, Predicted Profit: {predicted_profit:.4f}, Classification: {classification[symbol]}")

            return {
                'predicted_action': classification[symbol],  # 예측 결과
                'indicators': {  # 계산된 지표
                    'rsi': rsi,
                    'macd': macd,
                    'macd_signal': macd_signal,
                    'bollinger_upper': bollinger_upper,
                    'bollinger_lower': bollinger_lower
                }
            }

        except Exception as e:
            # print(f"{symbol} - Prediction Error: {e}")
            return {symbol: "예측 실패"}

    else:
        print(f"{symbol} - Indicator values contain None. Skipping.")
        return {symbol: "지표 계산 실패"}

