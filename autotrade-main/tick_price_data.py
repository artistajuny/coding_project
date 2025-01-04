import requests
import json
import datetime
import time
import yaml
import mysql.connector
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# config.yaml 파일 설정 로드
with open('D:\\coding_project\\autotrade-main\\config.yaml', encoding='UTF-8') as f:
    _cfg = yaml.load(f, Loader=yaml.FullLoader)

APP_KEY = _cfg['APP_KEY']
APP_SECRET = _cfg['APP_SECRET']
ACCESS_TOKEN = ""
URL_BASE = _cfg['URL_BASE']

# MySQL connection pooling 사용 설정
conn_pool = mysql.connector.pooling.MySQLConnectionPool(
    pool_name="mypool",
    pool_size=5,
    host=_cfg['DB_HOST'],
    user=_cfg['DB_USER'],
    password=_cfg['DB_PASS'],
    database=_cfg['DB_NAME']
)

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
        print(f"토큰 갱신 실패: {e}")
        raise e

def fetch_symbols():
    """stock_info 테이블에서 id가 449 이상인 모든 종목 코드 가져오기"""
    conn = conn_pool.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT symbol FROM stock_info WHERE id >= 449")
    symbols = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    return symbols

def fetch_trade_data(symbol, end_time):
    """특정 종목의 지정된 시간 이전의 체결 데이터를 조회"""
    PATH = "uapi/domestic-stock/v1/quotations/inquire-time-itemconclusion"
    URL = f"{URL_BASE}/{PATH}"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "FHPST01060000"
    }
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": symbol,
        "FID_INPUT_HOUR_1": end_time  # 조회 종료 시간
    }
    
    res = requests.get(URL, headers=headers, params=params)
    data = res.json()
    return data.get("output2", [])

def save_trade_data(symbol, trades):
    """체결 데이터를 stock_tick_data 테이블에 저장하고 200건을 초과하면 True 반환"""
    conn = conn_pool.get_connection()
    cursor = conn.cursor()
    insert_sql = """
    INSERT INTO stock_tick_data (symbol, trade_time, price, change_sign, 
                                 change_from_prev, change_rate, trade_volume, trade_strength) 
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        price = VALUES(price),
        change_sign = VALUES(change_sign),
        change_from_prev = VALUES(change_from_prev),
        change_rate = VALUES(change_rate),
        trade_volume = VALUES(trade_volume),
        trade_strength = VALUES(trade_strength)
    """
    for trade in trades:
        if int(trade['cnqn']) >= 10:
            values = (
                symbol,
                datetime.datetime.strptime(trade['stck_cntg_hour'], '%H%M%S').time(),
                float(trade['stck_prpr']),
                trade['prdy_vrss_sign'],
                float(trade['prdy_vrss']),
                float(trade['prdy_ctrt']),
                int(trade['cnqn']),
                float(trade['tday_rltv'])
            )
            cursor.execute(insert_sql, values)

    cursor.execute("SELECT COUNT(*) FROM stock_tick_data WHERE symbol = %s", (symbol,))
    row_count = cursor.fetchone()[0]
    conn.commit()
    cursor.close()
    conn.close()
    return row_count >= 200  # 200건 초과 시 True 반환

def process_symbol(symbol):
    """특정 종목을 3분 단위로 조회 및 저장"""
    start_time = "145900"
    while start_time >= "090000":
        intraday_data = fetch_trade_data(symbol, end_time=start_time)
        if intraday_data:
            stop_insertion = save_trade_data(symbol, intraday_data)
            print(f"{symbol}의 {start_time} 이전 체결 데이터 삽입 완료")
            if stop_insertion:
                print(f"{symbol}의 데이터가 200건을 초과하여 종료합니다.")
                break
        else:
            print(f"{symbol}의 {start_time} 이전 체결 데이터 없음")
        
        time_obj = datetime.datetime.strptime(start_time, "%H%M%S")
        time_obj -= datetime.timedelta(minutes=3)
        start_time = time_obj.strftime("%H%M%S")
        time.sleep(0.3)

# 모든 심볼에 대해 병렬 실행
try:
    ACCESS_TOKEN = get_access_token()
    symbols = fetch_symbols()

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_symbol, symbol) for symbol in symbols]
        for future in as_completed(futures):
            future.result()
except Exception as e:
    print(f"[오류 발생] {e}")
