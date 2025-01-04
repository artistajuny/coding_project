import yaml
import mysql.connector
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# config.yaml 파일 경로 업데이트
with open('D:\\coding_project\\autotrade-main\\config.yaml', encoding='UTF-8') as f:
    _cfg = yaml.load(f, Loader=yaml.FullLoader)

# MySQL connection pooling 사용 설정
conn_pool = mysql.connector.pooling.MySQLConnectionPool(
    pool_name="mypool",
    pool_size=5,
    host=_cfg['DB_HOST'],
    user=_cfg['DB_USER'],
    password=_cfg['DB_PASS'],
    database=_cfg['DB_NAME']
)

def calculate_rsi(closes, period=14):
    closes = [float(price) for price in closes]
    deltas = np.diff(closes)
    gains = deltas[deltas > 0].sum() / period
    losses = -deltas[deltas < 0].sum() / period
    rs = gains / losses if losses != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_moving_average(closes, period=20):
    return np.mean(closes[-period:])

def calculate_macd(closes, short_period=12, long_period=26, signal_period=9):
    short_ema = np.mean(closes[-short_period:])
    long_ema = np.mean(closes[-long_period:])
    macd = short_ema - long_ema
    signal = np.mean(closes[-signal_period:])
    return macd, signal

def calculate_bollinger_bands(closes, period=20, std_dev=2):
    sma = np.mean(closes[-period:])
    std_dev_value = np.std(closes[-period:])
    upper_band = sma + (std_dev_value * std_dev)
    lower_band = sma - (std_dev_value * std_dev)
    return upper_band, lower_band

def fetch_all_tick_data(symbol):
    conn = conn_pool.get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT timestamp, price FROM stock_tick_data WHERE symbol = %s ORDER BY timestamp DESC",
        (symbol,)
    )
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return [(row[0], row[1]) for row in data]

def batch_insert_tick_indicators(indicators, retries=3):
    conn = conn_pool.get_connection()
    cursor = conn.cursor()
    insert_sql = """
    INSERT INTO tick_indicators (symbol, timestamp, rsi, moving_average, macd, macd_signal, bollinger_upper, bollinger_lower)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        rsi = VALUES(rsi),
        moving_average = VALUES(moving_average),
        macd = VALUES(macd),
        macd_signal = VALUES(macd_signal),
        bollinger_upper = VALUES(bollinger_upper),
        bollinger_lower = VALUES(bollinger_lower)
    """
    indicators = [
        (symbol, timestamp, float(rsi), float(moving_average), float(macd), float(macd_signal), float(bollinger_upper), float(bollinger_lower))
        for symbol, timestamp, rsi, moving_average, macd, macd_signal, bollinger_upper, bollinger_lower in indicators
    ]
    for attempt in range(retries):
        try:
            cursor.executemany(insert_sql, indicators)
            conn.commit()
            break
        except mysql.connector.errors.InternalError as e:
            if "1213" in str(e):
                print(f"Deadlock 발생, {attempt + 1}번째 재시도 중...")
                time.sleep(1)
            else:
                raise e
    cursor.close()
    conn.close()

def process_symbol_tick_data(symbol):
    tick_data = fetch_all_tick_data(symbol)
    if len(tick_data) < 30:
        print(f"{symbol} 데이터가 부족하여 지표 계산을 건너뜁니다.")
        return

    indicators = []
    # 마지막 30건 제외하고 지표 계산
    for i in range(0, len(tick_data) - 30):
        timestamp, _ = tick_data[i]  # timestamp와 price로 분리
        closes = [price for _, price in tick_data[i:i + 30]]

        rsi = calculate_rsi(closes)
        moving_average = calculate_moving_average(closes)
        macd, macd_signal = calculate_macd(closes)
        bollinger_upper, bollinger_lower = calculate_bollinger_bands(closes)

        indicators.append((symbol, timestamp, rsi, moving_average, macd, macd_signal, bollinger_upper, bollinger_lower))

    batch_insert_tick_indicators(indicators)
    print(f"{symbol}의 지표 데이터가 저장되었습니다.")

def main():
    conn = conn_pool.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT symbol FROM stock_tick_data")
    symbols = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_symbol_tick_data, symbol) for symbol in symbols]
        for future in as_completed(futures):
            future.result()

if __name__ == "__main__":
    main()
