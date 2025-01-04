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
        autocommit = True  # autocommit을 연결 후에 설정합니다
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

def get_stock_info(symbol):
    """주식 기본정보 조회"""
    PATH = "uapi/domestic-stock/v1/quotations/search-stock-info"
    URL = f"{URL_BASE}/{PATH}"
    headers = {
        "Content-Type": "application/json", 
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "CTPF1002R",
        "custtype": "P",
    }
    params = {
        "PRDT_TYPE_CD": "300",
        "PDNO": symbol
    }
    res = requests.get(URL, headers=headers, params=params)
    info = res.json()['output']
    # send_message(f"주식 기본정보: {info}")
    return info

def update_stock_info_in_db(symbol, stock_data):
    """기존 stock_info 테이블에서 symbol에 해당하는 항목을 업데이트"""
    conn = get_db_connection()
    cursor = conn.cursor()
    # listing_date 변환 코드 추가
    # listing_date = stock_data.get('scts_mket_lstg_dt') ##코스피
    listing_date = stock_data.get('kosdaq_mket_lstg_dt')
    if listing_date:
        try:
            listing_date = datetime.datetime.strptime(listing_date, '%Y%m%d').date()
        except ValueError as e:
            send_message(f"날짜 변환 오류: {e}")
            return  # 날짜 변환 실패 시 업데이트를 건너뜀
    sql = """
    UPDATE stock_info
    SET name = %s, sector = %s, industry = %s, listing_date = %s
    WHERE symbol = %s
    """
    values = (
        stock_data['prdt_abrv_name'],  # 상품명
        stock_data.get('std_idst_clsf_cd_name'),  # 섹터
        stock_data.get('idx_bztp_scls_cd_name'),  # 산업
        listing_date,  # 상장일
        symbol
    )
    try:
        # 데이터 확인용 로그 출력
        send_message(f"Updating data: {values}")
        
        cursor.execute(sql, values)
        print(f"Updated rows: {cursor.rowcount}") 
        conn.commit()
    except mysql.connector.Error as e:
        # MySQL 에러 메시지 출력
        send_message(f"업데이트 실패: {e}")
    finally:
        cursor.close()
        conn.close()

def update_all_stocks():
    """DB에 있는 모든 종목의 기본 정보를 가져와 업데이트 (ID가 846번 이상인 경우만)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, symbol FROM stock_info WHERE id >= 846")
    symbols = cursor.fetchall()  # 이 부분에서 (id, symbol) 형태의 튜플 리스트로 가져옴
    cursor.close()
    conn.close()

    for id, symbol in symbols:
        try:
            stock_data = get_stock_info(symbol)
            update_stock_info_in_db(symbol, stock_data)  # DB에 저장 또는 업데이트
            send_message(f"{symbol} (ID: {id}) 정보 업데이트 완료")
            time.sleep(0.5)  # API 호출 간 딜레이
        except Exception as e:
            send_message(f"{symbol} (ID: {id}) 업데이트 실패: {e}")

# 자동매매 시작
try:
    ACCESS_TOKEN = get_access_token()
    get_stock_balance()
    get_balance()
    update_all_stocks()
   

except Exception as e:
    send_message(f"[오류 발생]{e}")
    time.sleep(1)
