import mysql.connector
import json
import yaml
import datetime

# config.yaml 파일 경로에서 DB 연결 정보 가져오기
with open('D:\\coding_project\\autotrade-main\\config.yaml', encoding='UTF-8') as f:
    _cfg = yaml.load(f, Loader=yaml.FullLoader)

def get_db_connection():
    """MySQL 데이터베이스 연결 생성"""
    conn = mysql.connector.connect(
        host=_cfg['DB_HOST'],
        user=_cfg['DB_USER'],
        password=_cfg['DB_PASS'],
        database=_cfg['DB_NAME']
    )
    return conn
def save_selected_stock(symbol, name, stotprice):
    """선별된 종목을 데이터베이스에 저장"""
    conn = get_db_connection()
    cursor = conn.cursor()
    sql = """
    INSERT INTO selected_stocks (symbol, name, stotprice, selected_date)
    VALUES (%s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE name=%s, stotprice=%s, selected_date=%s
    """
    cursor.execute(sql, (symbol, name, stotprice, datetime.datetime.now(), name, stotprice, datetime.datetime.now()))
    conn.commit()
    cursor.close()
    conn.close()

def save_stock_price(symbol, date, open_price, high, low, close, volume):
    """주식 가격을 데이터베이스에 저장"""
    conn = get_db_connection()
    cursor = conn.cursor()
    sql = """
    INSERT INTO stock_prices (symbol, date, open, high, low, close, volume)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE open=%s, high=%s, low=%s, close=%s, volume=%s
    """
    cursor.execute(sql, (symbol, date, open_price, high, low, close, volume,
                         open_price, high, low, close, volume))
    conn.commit()
    cursor.close()
    conn.close()
    
def save_trade_set_to_db(symbol, buy_price, sell_price, buy_time, sell_time, profit_loss, buy_conditions, sell_conditions, signals):
    """매수-매도 세트 데이터를 데이터베이스에 저장"""
    conn = get_db_connection()
    cursor = conn.cursor()
    sql = """
    INSERT INTO trade_sets (symbol, buy_price, sell_price, buy_time, sell_time, profit_loss, buy_conditions, sell_conditions, signals)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(sql, (symbol, buy_price, sell_price, buy_time, sell_time, profit_loss, json.dumps(buy_conditions), json.dumps(sell_conditions), json.dumps(signals)))
    conn.commit()
    cursor.close()
    conn.close()