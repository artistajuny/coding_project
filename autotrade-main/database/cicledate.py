import mysql.connector
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import yaml

with open('D:\\coding_project\\autotrade-main\\config.yaml', encoding='UTF-8') as f:
    _cfg = yaml.load(f, Loader=yaml.FullLoader)

def get_db_connection():
    """MySQL 데이터베이스 연결 생성"""
    return mysql.connector.connect(
        host=_cfg['DB_HOST'],
        user=_cfg['DB_USER'],
        password=_cfg['DB_PASS'],
        database=_cfg['DB_NAME'],
        autocommit=True
    )

def calculate_seasonal_component(stock_prices):
    """계절성 값 계산"""
    # stock_prices는 날짜별 close 값으로 정렬된 pandas Series
    stock_prices = stock_prices.interpolate(method='linear')  # NaN 값 보간
    decomposition = seasonal_decompose(stock_prices, model='additive', period=7, extrapolate_trend='freq')
    return decomposition.seasonal

def process_and_store_cycle_data(conn, batch_size=100):
    """주기 데이터 처리 및 저장"""
    cursor = conn.cursor(dictionary=True)
    
    # 종목과 날짜 가져오기
    cursor.execute("SELECT DISTINCT symbol FROM ai_tr_stock_price")
    symbols = [row['symbol'] for row in cursor.fetchall()]
    
    for symbol in symbols:
        cursor.execute(f"""
            SELECT date, close 
            FROM ai_tr_stock_price
            WHERE symbol = %s
            ORDER BY date ASC
        """, (symbol,))
        
        rows = cursor.fetchall()
        if not rows:
            continue
        
        # Pandas DataFrame으로 변환
        df = pd.DataFrame(rows)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # 계절성 값 계산
        seasonal = calculate_seasonal_component(df['close'])
        df['seasonal'] = seasonal
        
        # 주기 데이터 추가
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = (df.index.month - 1) // 3 + 1
        
        # 데이터 삽입
        insert_query = """
            INSERT INTO cycle_data (symbol, date, day_of_week, month, quarter, seasonal)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                day_of_week = VALUES(day_of_week),
                month = VALUES(month),
                quarter = VALUES(quarter),
                seasonal = VALUES(seasonal)
        """
        
        data_to_insert = [
            (symbol, date.strftime('%Y-%m-%d'), row['day_of_week'], row['month'], row['quarter'], row['seasonal'])
            for date, row in df.iterrows()
        ]
        
        for i in range(0, len(data_to_insert), batch_size):
            cursor.executemany(insert_query, data_to_insert[i:i + batch_size])
            conn.commit()
    
    cursor.close()

def main():
    conn = get_db_connection()
    try:
        print("Processing and storing cycle data...")
        process_and_store_cycle_data(conn)
        print("Cycle data processing complete!")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
