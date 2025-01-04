import requests
import json
import datetime
import time
import yaml
import os
import mysql.connector

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


def get_stock_price_by_date(symbol, date_str):
    """특정 날짜의 주가 정보 조회"""
    PATH = "uapi/domestic-stock/v1/quotations/inquire-daily-price"
    URL = f"{URL_BASE}/{PATH}"
    headers = {
        "Content-Type": "application/json", 
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "FHKST01010400",
        "custtype": "P",
    }
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": symbol,
        "FID_PERIOD_DIV_CODE": "D",
        "FID_ORG_ADJ_PRC": "1"
    }
    res = requests.get(URL, headers=headers, params=params)
    
    prices = res.json().get('output', [])
    # 지정한 날짜의 데이터만 필터링
    return [price for price in prices if price['stck_bsop_date'] == date_str]

def insert_stock_prices(conn, symbol, prices):
    """주가 정보를 stock_prices 테이블에 삽입"""
    cursor = conn.cursor()
    
    sql = """
    INSERT INTO stock_prices (symbol, date, open, high, low, close, volume) 
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        open = VALUES(open),
        high = VALUES(high),
        low = VALUES(low),
        close = VALUES(close),
        volume = VALUES(volume)
    """
    
    for price in prices:
        values = (
            symbol,
            datetime.datetime.strptime(price['stck_bsop_date'], '%Y%m%d').date(),
            float(price['stck_oprc']),
            float(price['stck_hgpr']),
            float(price['stck_lwpr']),
            float(price['stck_clpr']),
            int(price['acml_vol'])
        )
        try:
            cursor.execute(sql, values)
        except mysql.connector.Error as e:
            send_message(f"{symbol} 데이터 삽입 실패: {e}")
    
    conn.commit()
    cursor.close()

def update_stock_prices_by_date(date_str):
    """모든 종목의 특정 날짜 주가 정보를 업데이트"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT symbol FROM stock_info")
    symbols = [row[0] for row in cursor.fetchall()]
    cursor.close()

    for symbol in symbols:
        try:
            prices = get_stock_price_by_date(symbol, date_str)
            if prices:
                insert_stock_prices(conn, symbol, prices)
                send_message(f"{symbol}의 {date_str} 주가 데이터 삽입 완료")
            else:
                send_message(f"{symbol}의 {date_str} 주가 데이터 없음")
        except Exception as e:
            send_message(f"{symbol} 주가 데이터 업데이트 실패: {e}")

    conn.close()

# 자동매매 시작
try:
    ACCESS_TOKEN = get_access_token()
    update_stock_prices_by_date("20241213")  # 원하는 날짜로 데이터 갱신
except Exception as e:
    send_message(f"[오류 발생]{e}")
    time.sleep(1)
    
