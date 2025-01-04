import requests
import json
import datetime
import time
import yaml
import os
import mysql.connector
from portfolio_predictor import predict_all_portfolio  # portfolio_predictor 모듈에서 함수 가져오기
from portfolio_selector import select_portfolio  # select_portfolio 모듈에서 함수 가져오기
import pandas as pd

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
        autocommit=True
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
    """주식 잔고 조회"""
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
    
    try:
        res = requests.get(URL, headers=headers, params=params)
        res.raise_for_status()  # 요청 에러 발생 시 예외 처리
        data = res.json()
        
        stock_list = data.get('output1', [])  # 주식 잔고 데이터
        evaluation = data.get('output2', [])  # 평가 정보
        
        stock_dict = {}
        send_message(f"====주식 보유잔고====")
        
        for stock in stock_list:
            product_name = stock.get('prdt_abrv_name', '알 수 없음')
            holding_quantity = int(stock.get('hldg_qty', 0))
            
            if holding_quantity > 0:
                stock_dict[stock['pdno']] = holding_quantity
                send_message(f"{product_name}({stock['pdno']}): {holding_quantity}주")
                time.sleep(0.1)
        
        if evaluation:
            send_message(f"주식 평가 금액: {evaluation[0].get('scts_evlu_amt', '0')}원")
            send_message(f"평가 손익 합계: {evaluation[0].get('evlu_pfls_smtl_amt', '0')}원")
            send_message(f"총 평가 금액: {evaluation[0].get('tot_evlu_amt', '0')}원")
        else:
            send_message("평가 데이터가 없습니다.")
        
        send_message(f"=================")
        return stock_dict

    except requests.exceptions.RequestException as e:
        send_message(f"주식 잔고 조회 실패: {e}")
        return {}

def check_portfolio_yesterday():
    """어제 날짜 데이터가 포트폴리오에 있는지 확인"""
    conn = get_db_connection()
    cursor = conn.cursor()
    yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    cursor.execute("SELECT COUNT(*) FROM portfolio WHERE date = %s", (yesterday,))
    result = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return result > 0


def filter_portfolio_by_prediction():
    """포트폴리오 종목 중 상승 확률 70% 이상인 심볼만 필터링"""
    send_message("포트폴리오 예측을 시작합니다.")
    predictions = predict_all_portfolio("LSTM_CNN_Transformer")

    filtered_portfolio = {}
    for prediction in predictions:
        symbol = prediction.get('symbol')
        rise_prob = prediction.get('rise_prob')
        if rise_prob and rise_prob >= 0.75:  # 상승 확률이 75% 이상인 경우 
            filtered_portfolio[symbol] = True
            send_message(f"{symbol}: 상승 확률 {rise_prob * 100:.2f}% (필터링됨)")

    return filtered_portfolio
# 매수 완료된 종목을 추적할 집합 추가
completed_orders = set()

def get_recent_portfolio_predictions():
    """포트폴리오 예측을 실행하고 결과를 저장"""
    send_message("포트폴리오 예측을 시작합니다.")
    predictions = predict_all_portfolio("LSTM_CNN_Transformer")
    
    # 예측 결과 저장
    conn = get_db_connection()
    try:
        store_prediction_results(predictions, conn)
    finally:
        conn.close()

    for prediction in predictions:
        send_message(prediction)

def get_next_trading_day(latest_date):
    """포트폴리오에서 가장 최근 날짜를 기준으로 다음 거래일 계산"""
    if latest_date.weekday() == 4:  # 금요일이면
        next_day = latest_date + datetime.timedelta(days=3)
    else:  # 월~목이면
        next_day = latest_date + datetime.timedelta(days=1)
    return next_day

def store_prediction_results(predictions, conn):
    """예측 결과를 prediction_results 테이블에 저장 (상승 확률만 저장)"""
    cursor = conn.cursor()
    
    # 포트폴리오 테이블에서 가장 최근 날짜 가져오기
    cursor.execute("SELECT MAX(date) FROM portfolio")
    latest_date = cursor.fetchone()[0]

    if latest_date:
        next_day = get_next_trading_day(latest_date)
    else:
        next_day = datetime.datetime.now()

    next_day_str = next_day.strftime('%Y-%m-%d')

    # 해당 날짜에 이미 저장된 데이터가 있는지 확인
    cursor.execute("""
        SELECT COUNT(*) FROM prediction_results WHERE prediction_date = %s
    """, (next_day_str,))
    existing_count = cursor.fetchone()[0]

    if existing_count > 0:
        send_message(f"{next_day_str}에 대한 예측 결과가 이미 저장되어 있습니다. 저장을 건너뜁니다.")
        cursor.close()
        return  # 저장 건너뛰기

    # 상승 확률 75% 이상 필터링
    filtered_predictions = [p for p in predictions if float(p['rise_prob']) >= 0.75]

    insert_query = """
    INSERT INTO prediction_results (symbol, prediction_date, predicted_rise_prob)
    VALUES (%s, %s, %s)
    ON DUPLICATE KEY UPDATE
        predicted_rise_prob = VALUES(predicted_rise_prob)
    """

    for prediction in filtered_predictions:
        symbol = prediction['symbol']
        rise_prob = round(float(prediction['rise_prob']) * 100, 2)  # 상승 확률 소수점 2자리까지 저장
        cursor.execute(insert_query, (
            symbol,
            next_day_str,
            rise_prob
        ))
    
    conn.commit()
    cursor.close()
    send_message(f"{len(filtered_predictions)}개의 예측 결과가 저장되었습니다.")



def get_recent_portfolio_symbols():
    """포트폴리오 테이블에서 최근 날짜의 심볼을 가져와 dictionary에 저장"""
    portfolio_symbol_dict = {}
    try:
        engine = get_db_connection()  # 데이터베이스 연결
        latest_date_query = "SELECT MAX(date) FROM portfolio"
        latest_date = pd.read_sql(latest_date_query, engine).iloc[0, 0]

        if latest_date is None:
            print("포트폴리오 테이블에 데이터가 없습니다.")
            return {}

        symbols_query = f"SELECT DISTINCT symbol FROM portfolio WHERE date = '{latest_date}'"
        symbols = pd.read_sql(symbols_query, engine)['symbol'].tolist()
        
        # 심볼을 dictionary에 저장
        portfolio_symbol_dict = {symbol: True for symbol in symbols}
    except Exception as e:
        print(f"포트폴리오 데이터를 가져오는 중 오류 발생: {e}")
    
    return portfolio_symbol_dict

def check_stock_data_ready(latest_prediction_date):
    """주가 데이터가 예측 날짜에 대해 준비되었는지 확인"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 예측 날짜의 주가 데이터가 존재하는지 확인
    stock_data_query = """
    SELECT COUNT(*) FROM stock_prices
    WHERE date = %s
    """
    cursor.execute(stock_data_query, (latest_prediction_date,))
    stock_data_count = cursor.fetchone()[0]

    cursor.close()
    conn.close()

    # 예측 날짜의 데이터가 존재하지 않으면 False 반환
    return stock_data_count > 0

def update_actual_results():
    """장 종료 후 예측 결과를 업데이트"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # prediction_results 테이블에서 업데이트가 필요한 데이터 가져오기
    fetch_query = """
    SELECT symbol, prediction_date FROM prediction_results
    WHERE actual_result IS NULL
    """
    cursor.execute(fetch_query)
    results_to_update = cursor.fetchall()

    for symbol, prediction_date in results_to_update:
        # 예측 날짜의 전일 종가 데이터 가져오기
        prev_day = prediction_date - datetime.timedelta(days=1)
        prev_price_query = """
        SELECT close FROM stock_prices
        WHERE symbol = %s AND date = %s
        """
        cursor.execute(prev_price_query, (symbol, prev_day))
        prev_day_price = cursor.fetchone()

        # 예측 날짜의 종가 데이터 가져오기
        prediction_price_query = """
        SELECT close FROM stock_prices
        WHERE symbol = %s AND date = %s
        """
        cursor.execute(prediction_price_query, (symbol, prediction_date))
        prediction_day_price = cursor.fetchone()

        if prev_day_price and prediction_day_price:
            # 전일 종가와 예측 날짜의 종가 비교
            actual_result = 1 if prediction_day_price[0] > prev_day_price[0] else 0

            # actual_result와 is_correct 업데이트
            update_query = """
            UPDATE prediction_results
            SET actual_result = %s, is_correct = (predicted_rise_prob >= 75 AND %s = 1)
            WHERE symbol = %s AND prediction_date = %s
            """
            cursor.execute(update_query, (actual_result, actual_result, symbol, prediction_date))

    conn.commit()
    cursor.close()
    conn.close()
    send_message("예측 결과가 업데이트되었습니다.")



# 추가적으로 websocket 관련 부분 및 관련 함수 삭제
try:
    ACCESS_TOKEN = get_access_token()
    get_stock_balance()

    if not check_portfolio_yesterday():
        select_portfolio()
    else:
        send_message("어제 날짜 포트폴리오가 이미 존재합니다. 포트폴리오 업데이트를 생략합니다.")
    
    get_recent_portfolio_predictions()
    filtered_portfolio_dict = filter_portfolio_by_prediction()
 
 
 # 데이터 준비 상태 확인 후 실제 결과 업데이트
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(prediction_date) FROM prediction_results")
    latest_prediction_date = cursor.fetchone()[0]
    cursor.close()
    conn.close()

    if latest_prediction_date and check_stock_data_ready(latest_prediction_date):
        update_actual_results()
    else:
        send_message("주가 데이터가 준비되지 않았습니다. 업데이트를 나중에 다시 시도하세요.")

except Exception as e:
    send_message(f"[오류 발생] {e}")
    time.sleep(1)
