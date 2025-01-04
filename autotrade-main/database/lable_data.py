import mysql.connector
import datetime
import yaml

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
def update_labels_batch(conn, start_date=None, batch_size=100):
    """labels 테이블 업데이트"""
    cursor = conn.cursor(dictionary=True)
    offset = 0
    while True:
        # 배치로 symbol 가져오기
        cursor.execute(
            f"""
            SELECT DISTINCT symbol 
            FROM ai_tr_stock_price
            ORDER BY symbol
            LIMIT {batch_size} OFFSET {offset}
            """
        )
        symbols = cursor.fetchall()
        if not symbols:
            break  # 더 이상 데이터가 없으면 종료

        for row in symbols:
            symbol = row['symbol']
            try:
                cursor.execute(
                    f"""
                    INSERT INTO labels (symbol, date, price_change, classification)
                    SELECT 
                        a.symbol,
                        a.date,
                        ((a.close - b.close) / b.close) * 100 AS price_change,
                        CASE 
                            WHEN a.close > b.close THEN 1
                            ELSE 0
                        END AS classification
                    FROM 
                        ai_tr_stock_price a
                    JOIN 
                        ai_tr_stock_price b
                    ON 
                        a.symbol = b.symbol 
                        AND b.date = (
                            SELECT MAX(date) 
                            FROM ai_tr_stock_price 
                            WHERE symbol = a.symbol 
                            AND date < a.date
                        )
                    WHERE 
                        a.symbol = %s AND a.date >= %s
                    ON DUPLICATE KEY UPDATE
                        price_change = VALUES(price_change),
                        classification = VALUES(classification)
                    """,
                    (symbol, start_date)
                )
            except Exception as e:
                print(f"Error processing symbol {symbol}: {e}")
                continue
        offset += batch_size
        print(f"Batch {offset // batch_size} complete.")

    cursor.close()



def main():
    conn = get_db_connection()
    try:
        start_date = "2023-12-13"  # 특정 날짜부터 계산 시작
        print("Updating labels table...")
        update_labels_batch(conn, start_date=start_date)
        print("Update complete!")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
