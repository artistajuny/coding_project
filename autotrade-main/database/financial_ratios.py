import requests
import json
import datetime
import time
import yaml
import os
import mysql.connector
import math

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
# 각 비율 데이터를 요청하는 함수들
def get_financial_ratio_data(symbol):
    """재무비율 데이터를 요청하여 반환"""
    URL = f"{URL_BASE}/uapi/domestic-stock/v1/finance/financial-ratio"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "FHKST66430300",
        "custtype": "P"
    }
    params = {
        "FID_DIV_CLS_CODE": "0",
        "fid_cond_mrkt_div_code": "J",
        "fid_input_iscd": symbol
    }
    response = requests.get(URL, headers=headers, params=params)
    if response.status_code == 200:
        # print("Financial Ratio Data for symbol:", symbol)
        # print(response.json())  # 응답 전체 출력
        return response.json().get("output", [])
    else:
        print(f"Error fetching financial ratio for {symbol}: {response.text}")
        return []

def get_profitability_ratio_data(symbol):
    """수익성 비율 데이터를 요청하여 반환"""
    URL = f"{URL_BASE}/uapi/domestic-stock/v1/finance/profit-ratio"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "FHKST66430400",
        "custtype": "P"
    }
    params = {
        "fid_cond_mrkt_div_code": "J",
        "fid_input_iscd": symbol,
        "FID_DIV_CLS_CODE": "0"
    }
    response = requests.get(URL, headers=headers, params=params)
    if response.status_code == 200:
        # print("Profitability Ratio Data for symbol:", symbol)
        # print(response.json())  # 응답 전체 출력
        return response.json().get("output", [])
    else:
        print(f"Error fetching profitability ratio for {symbol}: {response.text}")
        return []

def get_other_major_ratios_data(symbol):
    """기타 주요 비율 데이터를 요청하여 반환"""
    URL = f"{URL_BASE}/uapi/domestic-stock/v1/finance/other-major-ratios"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "FHKST66430500",
        "custtype": "P"
    }
    params = {
        "fid_cond_mrkt_div_code": "J",
        "fid_input_iscd": symbol,
        "FID_DIV_CLS_CODE": "0"
    }
    response = requests.get(URL, headers=headers, params=params)
    if response.status_code == 200:
        # print("Other Major Ratios Data for symbol:", symbol)
        # print(response.json())  # 응답 전체 출력
        return response.json().get("output", [])
    else:
        print(f"Error fetching other major ratios for {symbol}: {response.text}")
        return []

def get_stability_ratio_data(symbol):
    """기타 안정성 비율 데이터를 요청하여 반환"""
    URL = f"{URL_BASE}/uapi/domestic-stock/v1/finance/stability-ratio"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "FHKST66430600",
        "custtype": "P"
    }
    params = {
        "fid_cond_mrkt_div_code": "J",
        "fid_input_iscd": symbol,
        "FID_DIV_CLS_CODE": "0"
    }
    response = requests.get(URL, headers=headers, params=params)
    if response.status_code == 200:
        # print("Stability Ratio Data for symbol:", symbol)
        # print(response.json())  # 응답 전체 출력
        return response.json().get("output", [])
    else:
        print(f"Error fetching stability ratio for {symbol}: {response.text}")
        return []

def get_growth_ratio_data(symbol):
    """기타 성장성 비율 데이터를 요청하여 반환"""
    URL = f"{URL_BASE}/uapi/domestic-stock/v1/finance/growth-ratio"
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "FHKST66430800",
        "custtype": "P"
    }
    params = {
        "fid_cond_mrkt_div_code": "J",
        "fid_input_iscd": symbol,
        "FID_DIV_CLS_CODE": "0"
    }
    response = requests.get(URL, headers=headers, params=params)
    if response.status_code == 200:
        # print("Growth Ratio Data for symbol:", symbol)
        # print(response.json())  # 응답 전체 출력
        return response.json().get("output", [])
    else:
        print(f"Error fetching growth ratio for {symbol}: {response.text}")
        return []
    
def sanitize_decimal(value):
    """무한대(inf), NaN 값을 NULL로 변환하고 소수점 2자리로 제한"""
    try:
        # 무한대 또는 NaN 체크 및 변환
        if value is None:
            return None
        if isinstance(value, (float, int)) and (math.isinf(value) or math.isnan(value)):
            return None
        # 소수점 2자리로 변환
        return round(float(value), 2)
    except (ValueError, TypeError):
        # 변환할 수 없는 경우 None 반환
        return None


def insert_financial_ratios(symbol, financial_ratio_data, profitability_ratio_data, other_major_ratios_data, stability_ratio_data, growth_ratio_data):
    """수집된 재무 비율 데이터를 financial_ratios 테이블에 삽입 (최근 5년 치 데이터만)"""
    conn = get_db_connection()
    cursor = conn.cursor()

    sql = """
    INSERT INTO financial_ratios (
        symbol, date, revenue_growth, operating_income_growth, net_income_growth, roe, eps, sps, bps, 
        reserve_ratio, debt_ratio, total_capital_profit_rate, self_capital_profit_rate,
        sales_net_profit_rate, sales_gross_profit_rate, payout_ratio, eva, ebitda, ev_ebitda, 
        borrowing_dependency, current_ratio, quick_ratio, equity_growth, total_assets_growth
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    ) ON DUPLICATE KEY UPDATE
        revenue_growth = VALUES(revenue_growth),
        operating_income_growth = VALUES(operating_income_growth),
        net_income_growth = VALUES(net_income_growth),
        roe = VALUES(roe),
        eps = VALUES(eps),
        sps = VALUES(sps),
        bps = VALUES(bps),
        reserve_ratio = VALUES(reserve_ratio),
        debt_ratio = VALUES(debt_ratio),
        total_capital_profit_rate = VALUES(total_capital_profit_rate),
        self_capital_profit_rate = VALUES(self_capital_profit_rate),
        sales_net_profit_rate = VALUES(sales_net_profit_rate),
        sales_gross_profit_rate = VALUES(sales_gross_profit_rate),
        payout_ratio = VALUES(payout_ratio),
        eva = VALUES(eva),
        ebitda = VALUES(ebitda),
        ev_ebitda = VALUES(ev_ebitda),
        borrowing_dependency = VALUES(borrowing_dependency),
        current_ratio = VALUES(current_ratio),
        quick_ratio = VALUES(quick_ratio),
        equity_growth = VALUES(equity_growth),
        total_assets_growth = VALUES(total_assets_growth)
    """

    # 5년 전 날짜 계산
    five_years_ago = datetime.datetime.now() - datetime.timedelta(days=5*365)

    for i in range(min(len(financial_ratio_data), len(profitability_ratio_data), 
                       len(other_major_ratios_data), len(stability_ratio_data), 
                       len(growth_ratio_data))):
        try:
            # 공통의 date 필드 가져오기
            date = datetime.datetime.strptime(financial_ratio_data[i]['stac_yymm'], '%Y%m').date()
            
            # 5년 이내의 데이터만 삽입
            if date >= five_years_ago.date():
                # 데이터 매핑 및 sanitize_decimal을 적용하여 무한대와 NaN 처리 및 소수점 2자리로 제한
                values = (
                    symbol,
                    date,
                    sanitize_decimal(financial_ratio_data[i].get("grs")),                         # revenue_growth
                    sanitize_decimal(financial_ratio_data[i].get("bsop_prfi_inrt")),              # operating_income_growth
                    sanitize_decimal(financial_ratio_data[i].get("ntin_inrt")),                   # net_income_growth
                    sanitize_decimal(financial_ratio_data[i].get("roe_val")),                     # roe
                    sanitize_decimal(financial_ratio_data[i].get("eps")),                         # eps
                    sanitize_decimal(financial_ratio_data[i].get("sps")),                         # sps
                    sanitize_decimal(financial_ratio_data[i].get("bps")),                         # bps
                    sanitize_decimal(financial_ratio_data[i].get("rsrv_rate")),                   # reserve_ratio
                    sanitize_decimal(financial_ratio_data[i].get("lblt_rate")),                   # debt_ratio
                    sanitize_decimal(profitability_ratio_data[i].get("cptl_ntin_rate")),          # total_capital_profit_rate
                    sanitize_decimal(profitability_ratio_data[i].get("self_cptl_ntin_inrt")),     # self_capital_profit_rate
                    sanitize_decimal(profitability_ratio_data[i].get("sale_ntin_rate")),          # sales_net_profit_rate
                    sanitize_decimal(profitability_ratio_data[i].get("sale_totl_rate")),          # sales_gross_profit_rate
                    sanitize_decimal(other_major_ratios_data[i].get("payout_rate")),              # payout_ratio
                    sanitize_decimal(other_major_ratios_data[i].get("eva")),                      # eva
                    sanitize_decimal(other_major_ratios_data[i].get("ebitda")),                   # ebitda
                    sanitize_decimal(other_major_ratios_data[i].get("ev_ebitda")),                # ev_ebitda
                    sanitize_decimal(stability_ratio_data[i].get("bram_depn")),                   # borrowing_dependency
                    sanitize_decimal(stability_ratio_data[i].get("crnt_rate")),                   # current_ratio
                    sanitize_decimal(stability_ratio_data[i].get("quck_rate")),                   # quick_ratio
                    sanitize_decimal(growth_ratio_data[i].get("equt_inrt")),                      # equity_growth
                    sanitize_decimal(growth_ratio_data[i].get("totl_aset_inrt"))                  # total_assets_growth
                )
                
                cursor.execute(sql, values)
        except KeyError as e:
            print(f"KeyError for symbol {symbol} at index {i}: Missing key {e}")
        except mysql.connector.Error as err:
            print(f"Error inserting financial ratios data for {symbol} on {date}: {err}")

    conn.commit()
    cursor.close()
    conn.close()



def update_financial_ratios_for_all():
    """모든 종목에 대해 재무 비율을 업데이트"""
    get_access_token()  # 토큰 갱신 체크

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT symbol FROM stock_info")
    symbols = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()

    for symbol in symbols:
        # 각각의 비율 데이터 요청
        financial_ratio_data = get_financial_ratio_data(symbol)
        profitability_ratio_data = get_profitability_ratio_data(symbol)
        other_major_ratios_data = get_other_major_ratios_data(symbol)
        stability_ratio_data = get_stability_ratio_data(symbol)
        growth_ratio_data = get_growth_ratio_data(symbol)

        # 각 데이터가 정상적으로 반환되었을 경우에만 저장
        if (financial_ratio_data and profitability_ratio_data and 
            other_major_ratios_data and stability_ratio_data and 
            growth_ratio_data):
            
            try:
                insert_financial_ratios(
                    symbol, financial_ratio_data, profitability_ratio_data, 
                    other_major_ratios_data, stability_ratio_data, growth_ratio_data
                )
                print(f"Inserted financial ratios data for {symbol}")
            except Exception as e:
                send_message(f"Error inserting data for {symbol}: {e}")
        else:
            send_message(f"No data for {symbol}")
        time.sleep(0.5)  # API 호출 간 딜레이


# 자동매매 시작
try:
    ACCESS_TOKEN = get_access_token()
    get_stock_balance()
    get_balance()
    update_financial_ratios_for_all()
   

except Exception as e:
    send_message(f"[오류 발생]{e}")
    time.sleep(1)
