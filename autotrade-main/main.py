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
from websocket_module import start_websocket, update_websocket_subscription  # 웹소켓 모듈에서 함수 가져오기

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
            # 'prdt_abrv_name' 또는 다른 필드가 없을 경우 기본값 설정
            product_name = stock.get('prdt_abrv_name', '알 수 없음')
            holding_quantity = int(stock.get('hldg_qty', 0))
            
            if holding_quantity > 0:
                stock_dict[stock['pdno']] = holding_quantity
                send_message(f"{product_name}({stock['pdno']}): {holding_quantity}주")
                time.sleep(0.1)
        
        # 평가 데이터 출력
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
    except KeyError as e:
        send_message(f"응답 데이터에 키가 없습니다: {e}")
        return {}
    except Exception as e:
        send_message(f"예기치 않은 오류 발생: {e}")
        return {}


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


# 포트폴리오 종목 예측
def get_recent_portfolio_predictions():
    send_message("포트폴리오 예측을 시작합니다.")
    predictions = predict_all_portfolio("LSTM_CNN_Transformer")
    for prediction in predictions:
        send_message(prediction)

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
    
# 실시간 데이터 수신 시작
def start_realtime_data(symbols):
    for symbol in symbols:
        start_websocket(symbol)

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
        if rise_prob and rise_prob >= 0.75:  # 상승 확률이 65% 이상인 경우 
            filtered_portfolio[symbol] = True
            send_message(f"{symbol}: 상승 확률 {rise_prob * 100:.2f}% (필터링됨)")

    return filtered_portfolio
# 매수 완료된 종목을 추적할 집합 추가
completed_orders = set()

def send_buy_order(symbol, trade_data, filtered_portfolio_dict, ws, retry_count=3):
    global completed_orders

    if symbol in completed_orders:
        print(f"[중복 방지] {symbol}은 이미 매수 완료된 종목입니다.")
        return

    current_price = trade_data.get('current_price')
    indicators = trade_data.get('indicators')
    predicted_action = trade_data.get('predicted_action')

    if not evaluate_buy_condition(symbol, current_price, indicators, predicted_action):
        print(f"[매수 조건 미충족] 종목: {symbol}")
        return

    for attempt in range(retry_count):
        try:
            PATH = "uapi/domestic-stock/v1/trading/order-cash"
            URL = f"{URL_BASE}/{PATH}"
            headers = {
                "Content-Type": "application/json",
                "authorization": f"Bearer {ACCESS_TOKEN}",
                "appKey": APP_KEY,
                "appSecret": APP_SECRET,
                "tr_id": "TTTC0802U",
                "custtype": "P",
            }
            body = {
                "CANO": CANO,
                "ACNT_PRDT_CD": ACNT_PRDT_CD,
                "PDNO": symbol,
                "ORD_DVSN": "00",
                "ORD_QTY": "1",
                "ORD_UNPR": str(current_price)
            }
            res = requests.post(URL, headers=headers, data=json.dumps(body))
            res.raise_for_status()

            send_message(f"매수 주문 성공: {symbol} 가격 {current_price}원")

            if symbol in filtered_portfolio_dict:
                del filtered_portfolio_dict[symbol]
                send_message(f"{symbol}이(가) filtered_portfolio_dict에서 삭제되었습니다.")
                print(f"삭제 후 딕셔너리 상태: {filtered_portfolio_dict}")
                update_websocket_subscription(ws, symbol, "remove")

            # 매수 완료된 종목으로 추가
            completed_orders.add(symbol)
            print(f"[추가됨] 매수 완료 종목: {completed_orders}")
            return

        except requests.exceptions.RequestException as e:
            error_msg = f"매수 주문 실패: {symbol} - {e.response.text if e.response else str(e)}"
            print(error_msg)
            send_message(error_msg)

            if attempt == retry_count - 1:
                print(f"[매수 주문 재시도 실패] {symbol} (총 {retry_count}회 시도)")
                send_message(f"[매수 주문 재시도 실패] {symbol} (총 {retry_count}회 시도)")
            else:
                time.sleep(1)  # 재시도 전에 대기


def send_sell_order(symbol, price):
    """매도 주문 실행"""
    try:
        PATH = "uapi/domestic-stock/v1/trading/order-cash"
        URL = f"{URL_BASE}/{PATH}"
        headers = {
            "Content-Type": "application/json",
            "authorization": f"Bearer {ACCESS_TOKEN}",
            "appKey": APP_KEY,
            "appSecret": APP_SECRET,
            "tr_id": "TTTC0801U",  # 매도 거래 ID
            "custtype": "P",
        }
        body = {
            "CANO": CANO,
            "ACNT_PRDT_CD": ACNT_PRDT_CD,
            "PDNO": symbol,
            "ORD_DVSN": "00",  # 시장가 주문
            "ORD_QTY": "1",   # 매도 수량 (예: 1주)
            "ORD_UNPR": price  # 매도 가격
        }
        res = requests.post(URL, headers=headers, data=json.dumps(body))
        res.raise_for_status()
        send_message(f"매도 주문 성공: {symbol} 가격 {price}원")
    except Exception as e:
        send_message(f"매도 주문 실패: {symbol} - {e}")


# 실시간 데이터 수신 콜백 함수
def signal_callback(trade_data, ws):
    """실시간 데이터 콜백 처리"""
    global filtered_portfolio_dict

    symbol = trade_data.get('symbol')
    current_price = trade_data.get('current_price')
    indicators = trade_data.get('indicators')  # 지표 데이터
    predicted_action = trade_data.get('predicted_action')  # AI 예측 결과

    # 데이터가 충분히 쌓이지 않아 지표가 None인 경우 처리
    if indicators is None:
        print(f"[지표 데이터 부족] 종목: {symbol}, 데이터가 충분히 쌓이지 않았습니다.")
        return

    # AI 예측 결과가 None인 경우 처리
    if predicted_action is None:
        print(f"[AI 예측 결과 부족] 종목: {symbol}, 예측 데이터가 없습니다.")
        return

    print(f"[지표 데이터] {indicators}")
    print(f"[AI 예측 결과] {predicted_action}")

    # 매수/매도 로직
    if evaluate_buy_condition(symbol, current_price, indicators, predicted_action):
        send_buy_order(symbol, trade_data, filtered_portfolio_dict, ws)
    elif evaluate_sell_condition(symbol, current_price, indicators, predicted_action):
        print(f"[매도 시그널 감지] 종목: {symbol}, 현재가: {current_price}")



def evaluate_buy_condition(symbol, current_price, indicators, predicted_action):
    """매수 조건 평가"""
    rsi = indicators.get('rsi')
    macd = indicators.get('macd')
    macd_signal = indicators.get('macd_signal')

    # 기본 조건: RSI가 과매수 영역이 아니고, MACD > MACD Signal
    basic_condition = (
        rsi is not None and macd is not None and macd_signal is not None and
        rsi < 70 and macd > macd_signal
    )

    # AI 예측 결과를 고려한 조건
    prediction_condition = predicted_action in ["중립","매수", "강력 매수"]

    return basic_condition or prediction_condition

def evaluate_sell_condition(symbol, current_price, change_rate, predicted_action):
    """매도 조건 평가"""
    # 예측 결과가 없으면 기본 조건만 평가
    if predicted_action is None:
        return change_rate < -3

    # 예측 결과가 있는 경우 조건 추가
    return (
        predicted_action in ["매도", "강력 매도"] or
        change_rate < -3
    )
def initialize_websocket(filtered_portfolio_dict):
    """웹소켓 초기화"""
    def websocket_callback(trade_data, ws):
        """웹소켓 데이터를 처리하는 콜백 함수"""
        try:
            signal_callback(trade_data, ws)  # ws 포함하여 signal_callback 호출
        except Exception as e:
            print(f"콜백 처리 중 오류: {e}")
            send_message(f"웹소켓 콜백 오류: {e}")

    # 웹소켓 시작
    ws = start_websocket(
        list(filtered_portfolio_dict.keys()),
        callback=websocket_callback  # 수정된 콜백 전달
    )
    return ws



try:
    ACCESS_TOKEN = get_access_token()
    get_stock_balance()
    get_balance()

    # 어제 날짜로 포트폴리오가 이미 구성되었는지 확인
    if not check_portfolio_yesterday():
        select_portfolio()
    else:
        send_message("어제 날짜 포트폴리오가 이미 존재합니다. 포트폴리오 업데이트를 생략합니다.")
    
    get_recent_portfolio_predictions()
    filtered_portfolio_dict = filter_portfolio_by_prediction()

    # 필터링된 심볼로 웹소켓 시작
    ws = initialize_websocket(filtered_portfolio_dict)

except Exception as e:
    send_message(f"[오류 발생] {e}")
    time.sleep(1)
