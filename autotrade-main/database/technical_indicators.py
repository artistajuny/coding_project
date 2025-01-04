import requests
import json
import datetime
import time
import yaml
import os
import mysql.connector
import numpy as np


# config.yaml 파일 경로 업데이트
with open('D:\\coding_project\\autotrade-main\\config.yaml', encoding='UTF-8') as f:
    _cfg = yaml.load(f, Loader=yaml.FullLoader)

APP_KEY = _cfg['APP_KEY']
APP_SECRET = _cfg['APP_SECRET']
ACCESS_TOKEN = ""
CANO = _cfg['CANO']
ACNT_PRDT_CD = _cfg['ACNT_PRDT_CD']
DISCORD_WEBHOOK_URL = _cfg['DISCORD_WEBHOOK_URL']
URL_BASE = _cfg['URL_BASE']
USER_ID = _cfg['USER_ID']

def get_db_connection():
    """MySQL 데이터베이스 연결 생성"""
    conn = mysql.connector.connect(
        host=_cfg['DB_HOST'],
        user=_cfg['DB_USER'],
        password=_cfg['DB_PASS'],
        database=_cfg['DB_NAME'],
        autocommit=True  # autocommit을 연결 후에 설정합니다
    )
    return conn

def send_message(msg):
    """디스코드 메세지 전송"""
    now = datetime.datetime.now()
    message = {"content": f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] {str(msg)}"}
    requests.post(DISCORD_WEBHOOK_URL, data=message)
    print(message)

def get_access_token():
    """토큰 발급 또는 파일에서 읽기"""
    global ACCESS_TOKEN
    token_file = "access_token.json"
    if os.path.exists(token_file):
        with open(token_file, 'r') as f:
            token_data = json.load(f)
            expiration_time = datetime.datetime.fromisoformat(token_data['expiration_time'])
            if datetime.datetime.now() < expiration_time:
                ACCESS_TOKEN = token_data['access_token']
                return ACCESS_TOKEN
    return refresh_access_token()

def refresh_access_token():
    """토큰 갱신"""
    global ACCESS_TOKEN
    headers = {"content-type": "application/json"}
    body = {
        "grant_type": "client_credentials",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET
    }
    PATH = "oauth2/tokenP"
    URL = f"{URL_BASE}/{PATH}"
    try:
        res = requests.post(URL, headers=headers, data=json.dumps(body))
        res.raise_for_status()
        
        access_token = res.json()["access_token"]
        expiration_time = datetime.datetime.now() + datetime.timedelta(hours=24)
        with open("access_token.json", 'w') as f:
            json.dump({
                'access_token': access_token,
                'expiration_time': expiration_time.isoformat()
            }, f)
        ACCESS_TOKEN = access_token
        return ACCESS_TOKEN
    except requests.exceptions.RequestException as e:
        send_message(f"토큰 갱신 실패: {e}")
        raise e
def get_stock_balance():
    """주식 잔고조회"""
    PATH = "uapi/domestic-stock/v1/trading/inquire-balance"
    URL = f"{URL_BASE}/{PATH}"
    headers = {
        "Content-Type": "application/json", 
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "TTTC8434R",
        "custtype": "P",
    }
    params = {
        "CANO": CANO,
        "ACNT_PRDT_CD": ACNT_PRDT_CD,
        "AFHR_FLPR_YN": "N",
        "OFL_YN": "",
        "INQR_DVSN": "02",
        "UNPR_DVSN": "01",
        "FUND_STTL_ICLD_YN": "N",
        "FNCG_AMT_AUTO_RDPT_YN": "N",
        "PRCS_DVSN": "01",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": ""
    }
    res = requests.get(URL, headers=headers, params=params)
    stock_list = res.json()['output1']
    evaluation = res.json()['output2']
    stock_dict = {}
    send_message(f"====주식 보유잔고====")
    for stock in stock_list:
        if int(stock['hldg_qty']) > 0:
            stock_dict[stock['pdno']] = stock['hldg_qty']
            send_message(f"{stock['prdt_abrv_name']}({stock['pdno']}): {stock['hldg_qty']}주")
            time.sleep(0.1)
    send_message(f"주식 평가 금액: {evaluation[0]['scts_evlu_amt']}원")
    time.sleep(0.1)
    send_message(f"평가 손익 합계: {evaluation[0]['evlu_pfls_smtl_amt']}원")
    time.sleep(0.1)
    send_message(f"총 평가 금액: {evaluation[0]['tot_evlu_amt']}원")
    time.sleep(0.1)
    send_message(f"=================")
    return stock_dict

def get_balance():
    """현금 잔고조회"""
    PATH = "uapi/domestic-stock/v1/trading/inquire-psbl-order"
    URL = f"{URL_BASE}/{PATH}"
    headers = {
        "Content-Type": "application/json", 
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "TTTC8908R",
        "custtype": "P",
    }
    params = {
        "CANO": CANO,
        "ACNT_PRDT_CD": ACNT_PRDT_CD,
        "PDNO": "005930",
        "ORD_UNPR": "65500",
        "ORD_DVSN": "01",
        "CMA_EVLU_AMT_ICLD_YN": "Y",
        "OVRS_ICLD_YN": "Y"
    }
    res = requests.get(URL, headers=headers, params=params)
    cash = res.json()['output']['ord_psbl_cash']
    send_message(f"주문 가능 현금 잔고: {cash}원")
    return int(cash)

def calculate_moving_average(closes, period=20):
    """이동평균 계산"""
    return np.mean(closes[-period:]) if len(closes) >= period else None

def calculate_rsi(closes, period=14):
    """RSI 계산"""
    gains = [max(0, closes[i] - closes[i - 1]) for i in range(1, len(closes))]
    losses = [abs(min(0, closes[i] - closes[i - 1])) for i in range(1, len(closes))]
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    return 100 - (100 / (1 + (avg_gain / avg_loss))) if avg_loss != 0 else 100

def calculate_macd(closes, short_period=12, long_period=26, signal_period=9):
    """MACD 계산"""
    if len(closes) < long_period:
        return None, None
    short_ema = np.mean(closes[-short_period:])
    long_ema = np.mean(closes[-long_period:])
    macd = short_ema - long_ema
    signal_line = np.mean(closes[-signal_period:]) if len(closes) >= signal_period else None
    return macd, signal_line

def calculate_bollinger_bands(closes, period=20, std_dev=2):
    """볼린저 밴드 계산"""
    if len(closes) < period:
        return None, None
    sma = np.mean(closes[-period:])
    std_dev_value = np.std(closes[-period:])
    upper_band = sma + (std_dev_value * std_dev)
    lower_band = sma - (std_dev_value * std_dev)
    return upper_band, lower_band

def calculate_stochastic_oscillator(closes, lows, highs, period=14):
    """스토캐스틱 오실레이터 계산"""
    if len(closes) < period:
        return None, None
    high_max = max(highs[-period:])
    low_min = min(lows[-period:])
    k = ((closes[-1] - low_min) / (high_max - low_min)) * 100 if high_max - low_min != 0 else 0
    d = np.mean([k])  # 보통은 3일 이동평균으로 계산
    return k, d

def calculate_atr(highs, lows, closes, period=14):
    """ATR 계산"""
    if len(closes) < period:
        return None
    tr = [max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])) for i in range(1, len(closes))]
    return np.mean(tr[-period:])

def calculate_cci(closes, highs, lows, period=20):
    """CCI 계산"""
    if len(closes) < period:
        return None
    tp = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(len(closes))]
    sma_tp = np.mean(tp[-period:])
    mean_dev = np.mean([abs(tp[i] - sma_tp) for i in range(-period, 0)])
    return (tp[-1] - sma_tp) / (0.015 * mean_dev) if mean_dev != 0 else None

def calculate_obv(closes, volumes):
    """OBV 계산"""
    obv = 0
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv += volumes[i]
        elif closes[i] < closes[i - 1]:
            obv -= volumes[i]
    return obv

def get_recent_stock_prices(symbol, days=30, conn=None):
    """특정 종목의 최근 주가 데이터 조회"""
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT date, close, open, high, low, volume FROM stock_prices WHERE symbol = %s ORDER BY date DESC LIMIT %s",
        (symbol, days)
    )
    prices = cursor.fetchall()
    cursor.close()

    # Decimal 값을 float으로 변환
    for price in prices:
        price['close'] = float(price['close'])
        price['open'] = float(price['open'])
        price['high'] = float(price['high'])
        price['low'] = float(price['low'])
        price['volume'] = float(price['volume'])

    return prices[::-1]

# 기술적 지표 계산 함수들 (각 calculate_* 함수 그대로 유지)

def insert_technical_indicators(symbol, date, moving_average, rsi, macd, macd_signal, bollinger_upper, bollinger_lower, stochastic_k, stochastic_d, atr, cci, obv, conn):
    """계산된 기술적 지표를 DB에 삽입"""
    cursor = conn.cursor()
    sql = """
    INSERT INTO technical_indicators (
        symbol, date, moving_average, rsi, macd, macd_signal, bollinger_upper, bollinger_lower,
        stochastic_k, stochastic_d, atr, cci, obv
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        moving_average = VALUES(moving_average),
        rsi = VALUES(rsi),
        macd = VALUES(macd),
        macd_signal = VALUES(macd_signal),
        bollinger_upper = VALUES(bollinger_upper),
        bollinger_lower = VALUES(bollinger_lower),
        stochastic_k = VALUES(stochastic_k),
        stochastic_d = VALUES(stochastic_d),
        atr = VALUES(atr),
        cci = VALUES(cci),
        obv = VALUES(obv)
    """
    cursor.execute(sql, (symbol, date, moving_average, rsi, macd, macd_signal, bollinger_upper, bollinger_lower, stochastic_k, stochastic_d, atr, cci, obv))
    conn.commit()
    cursor.close()

def get_recent_stock_prices(symbol, days=30, conn=None):
    """특정 종목의 최근 주가 데이터 조회"""
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT date, close, open, high, low, volume FROM stock_prices WHERE symbol = %s ORDER BY date DESC LIMIT %s",
        (symbol, days)
    )
    prices = cursor.fetchall()
    cursor.close()

    # Decimal 값을 float으로 변환
    for price in prices:
        price['close'] = float(price['close'])
        price['open'] = float(price['open'])
        price['high'] = float(price['high'])
        price['low'] = float(price['low'])
        price['volume'] = float(price['volume'])

    return prices[::-1]

# 기술적 지표 계산 함수들 (각 calculate_* 함수 그대로 유지)

def insert_technical_indicators(symbol, date, moving_average, rsi, macd, macd_signal, bollinger_upper, bollinger_lower, stochastic_k, stochastic_d, atr, cci, obv, conn):
    """계산된 기술적 지표를 DB에 삽입"""
    # 모든 float64 타입의 값을 float으로 변환
    moving_average = float(moving_average) if moving_average is not None else None
    rsi = float(rsi) if rsi is not None else None
    macd = float(macd) if macd is not None else None
    macd_signal = float(macd_signal) if macd_signal is not None else None
    bollinger_upper = float(bollinger_upper) if bollinger_upper is not None else None
    bollinger_lower = float(bollinger_lower) if bollinger_lower is not None else None
    stochastic_k = float(stochastic_k) if stochastic_k is not None else None
    stochastic_d = float(stochastic_d) if stochastic_d is not None else None
    atr = float(atr) if atr is not None else None
    cci = float(cci) if cci is not None else None
    obv = float(obv) if obv is not None else None

    cursor = conn.cursor()
    sql = """
    INSERT INTO technical_indicators (
        symbol, date, moving_average, rsi, macd, macd_signal, bollinger_upper, bollinger_lower,
        stochastic_k, stochastic_d, atr, cci, obv
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        moving_average = VALUES(moving_average),
        rsi = VALUES(rsi),
        macd = VALUES(macd),
        macd_signal = VALUES(macd_signal),
        bollinger_upper = VALUES(bollinger_upper),
        bollinger_lower = VALUES(bollinger_lower),
        stochastic_k = VALUES(stochastic_k),
        stochastic_d = VALUES(stochastic_d),
        atr = VALUES(atr),
        cci = VALUES(cci),
        obv = VALUES(obv)
    """
    cursor.execute(sql, (symbol, date, moving_average, rsi, macd, macd_signal, bollinger_upper, bollinger_lower, stochastic_k, stochastic_d, atr, cci, obv))
    conn.commit()
    cursor.close()

def update_technical_indicators(symbol, conn):
    """특정 종목에 대한 최신 기술적 지표 업데이트"""
    prices = get_recent_stock_prices(symbol, conn=conn)
    if not prices:
        send_message(f"{symbol} 주가 데이터가 부족하여 지표 계산을 건너뜁니다.")
        return

    closes = [price['close'] for price in prices]
    highs = [price['high'] for price in prices]
    lows = [price['low'] for price in prices]
    volumes = [price['volume'] for price in prices]
    date = prices[-1]['date']

    # 지표 계산
    moving_average = calculate_moving_average(closes)
    rsi = calculate_rsi(closes)
    macd, macd_signal = calculate_macd(closes)
    bollinger_upper, bollinger_lower = calculate_bollinger_bands(closes)
    stochastic_k, stochastic_d = calculate_stochastic_oscillator(closes, lows, highs)
    atr = calculate_atr(highs, lows, closes)
    cci = calculate_cci(closes, highs, lows)
    obv = calculate_obv(closes, volumes)

    # 계산된 지표를 DB에 삽입
    insert_technical_indicators(
        symbol, date, moving_average, rsi, macd, macd_signal,
        bollinger_upper, bollinger_lower, stochastic_k, stochastic_d,
        atr, cci, obv, conn
    )
    send_message(f"{symbol}의 기술적 지표 계산 및 DB 삽입 완료")

def update_all_technical_indicators():
    """모든 종목의 기술적 지표 업데이트"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT symbol FROM stock_info")
    symbols = [row[0] for row in cursor.fetchall()]
    cursor.close()
    
    for symbol in symbols:
        try:
            update_technical_indicators(symbol, conn)
        except Exception as e:
            send_message(f"{symbol} 지표 업데이트 실패: {e}")
    conn.close()

# 자동매매 시작
try:
    ACCESS_TOKEN = get_access_token()
    update_all_technical_indicators()

except Exception as e:
    send_message(f"[오류 발생] {e}")
    time.sleep(1)