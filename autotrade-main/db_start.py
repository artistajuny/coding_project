import pandas as pd
import mysql.connector
import yaml

# config.yaml 파일 경로 업데이트
with open('D:\\coding_project\\autotrade-main\\config.yaml', encoding='utf-8') as f:
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

# CSV 파일 로드
df = pd.read_csv('D:\\coding_project\\autotrade-main\\kosdaq.csv', encoding='utf-8')

# 필요한 열만 선택하고 데이터 정리
data = df[['단축코드', '한글 종목약명']].rename(columns={'단축코드': 'symbol', '한글 종목약명': 'name'})

# 데이터베이스에 삽입
def insert_stock_info(data):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    sql = """
    INSERT INTO stock_info (symbol, name, exchange) 
    VALUES (%s, %s, %s) 
    ON DUPLICATE KEY UPDATE name=%s, exchange=%s
    """
    
    for _, row in data.iterrows():
        # symbol 값이 6자리가 되도록 앞에 0을 붙여줌
        symbol = str(row['symbol']).zfill(6)
        cursor.execute(sql, (symbol, row['name'], "KOSDAQ", row['name'], "KOSDAQ"))
    
    conn.commit()
    cursor.close()
    conn.close()

insert_stock_info(data)
print("데이터 삽입 완료")
