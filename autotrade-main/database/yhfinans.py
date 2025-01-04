import yfinance as yf
import mysql.connector
from datetime import datetime
import pandas as pd
import yaml

# Config 파일 로드
with open('D:\\coding_project\\autotrade-main\\config.yaml', encoding='UTF-8') as f:
    _cfg = yaml.load(f, Loader=yaml.FullLoader)

def get_db_connection():
    """MySQL 데이터베이스 연결 생성"""
    return mysql.connector.connect(
        host=_cfg['DB_HOST'],
        user=_cfg['DB_USER'],
        password=_cfg['DB_PASS'],
        database=_cfg['DB_NAME']
    )
def fetch_yahoo_finance_data_range(ticker, date):
    """Yahoo Finance에서 특정 날짜의 데이터를 가져오기"""
    try:
        data = yf.Ticker(ticker)
        # 날짜 범위를 단일 날짜로 설정
        hist = data.history(start=date, end=(pd.to_datetime(date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
        if not hist.empty:
            return hist['Close']  # 날짜별 종가 반환
        else:
            # 데이터가 비어 있으면 휴장일로 간주
            print(f"No data found for {ticker} on {date}. Possibly a holiday or weekend.")
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
    return None

def store_external_factors(conn, date, factors):
    """외부 요인 데이터를 MySQL에 저장"""
    # float64를 Python 기본 float으로 변환
    factors = [float(factor) if factor is not None else None for factor in factors]
    cursor = conn.cursor()
    query = """
        INSERT INTO external_factors (
            date, kospi, kosdaq, s_and_p_500, oil_price, gold_price, vix, usd_krw, eur_usd, nasdaq
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            kospi = VALUES(kospi),
            kosdaq = VALUES(kosdaq),
            s_and_p_500 = VALUES(s_and_p_500),
            oil_price = VALUES(oil_price),
            gold_price = VALUES(gold_price),
            vix = VALUES(vix),
            usd_krw = VALUES(usd_krw),
            eur_usd = VALUES(eur_usd),
            nasdaq = VALUES(nasdaq)
    """
    cursor.execute(query, (date, *factors))
    conn.commit()
    cursor.close()
def main():
    conn = get_db_connection()
    try:
        print("Fetching and storing external factors...")

        # 기간 설정
        start_date = "2023-12-13"
        end_date = "2024-12-13"

        # Yahoo Finance 데이터를 가져올 심볼들
        tickers = {
            "kospi": "^KS11",        # KOSPI 지수
            "kosdaq": "^KQ11",       # KOSDAQ 지수
            "s_and_p_500": "^GSPC",  # S&P 500 지수
            "oil_price": "CL=F",     # WTI 원유 가격
            "gold_price": "GC=F",    # 금 선물 가격
            "vix": "^VIX",           # 변동성 지수
            "usd_krw": "KRW=X",      # USD/KRW 환율
            "eur_usd": "EURUSD=X",   # EUR/USD 환율
            "nasdaq": "^IXIC"        # NASDAQ 지수
        }

        # 날짜 범위 생성
        date_range = pd.date_range(start=start_date, end=end_date)

        for current_date in date_range:
            formatted_date = current_date.strftime("%Y-%m-%d")

            # 각 날짜별 데이터 수집
            factors = {}
            skip_date = False
            for key, value in tickers.items():
                result = fetch_yahoo_finance_data_range(value, formatted_date)
                if result is None or result.empty:
                    skip_date = True  # 지수 값이 없는 경우 건너뜀
                    break
                factors[key] = result.iloc[-1] if result is not None else None

            if skip_date:
                print(f"Skipping {formatted_date}: No index data available (possibly holiday/weekend).")
                continue

            # 데이터를 저장할 준비
            factor_values = [
                factors.get("kospi"),
                factors.get("kosdaq"),
                factors.get("s_and_p_500"),
                factors.get("oil_price"),
                factors.get("gold_price"),
                factors.get("vix"),
                factors.get("usd_krw"),
                factors.get("eur_usd"),
                factors.get("nasdaq")
            ]

            # None 값 처리 및 float 변환
            factor_values = [
                float(val) if pd.notna(val) else None for val in factor_values
            ]

            # 데이터 저장
            store_external_factors(conn, formatted_date, factor_values)

        print("External factors updated successfully!")
    finally:
        conn.close()

if __name__ == "__main__":
    main()