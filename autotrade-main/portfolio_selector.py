import mysql.connector
import yaml
from datetime import datetime

# config.yaml 파일 경로 업데이트
with open('D:\\coding_project\\autotrade-main\\config.yaml', encoding='UTF-8') as f:
    _cfg = yaml.load(f, Loader=yaml.FullLoader)

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

def select_portfolio():
    """포트폴리오 종목 선별 및 데이터베이스 업데이트"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # 예외 처리할 종목 리스트
    exclude_symbols = [
        '025950', '045660', '224110', '065500', '093240', '045340', '121850', '065770', '204840',
        '050110', '024070', '050960', '105840', '236200', '208140', '002070', '130740', '214680',
        '258790', '002720', '001840', '011080', '015020', '002410',
        '007860', '004830', '004540', '014440',
        '053800', '004770', '049480', '013700', '053050', '093640', '025560',
        '008350', '026940', '091590', '084680', '104480',
        '007770', '010640', '003780', '051630'
    ]

    # 가장 최근 거래일과 그 이전 거래일 가져오기
    cursor.execute("""
        SELECT DISTINCT date 
        FROM stock_prices 
        ORDER BY date DESC 
        LIMIT 2
    """)
    dates = cursor.fetchall()
    if len(dates) < 2:
        print("충분한 데이터가 없습니다.")
        return

    latest_date = dates[0]['date']
    previous_date = dates[1]['date']

    # 3%~15% 상승한 종목 선택
    cursor.execute(f"""
        SELECT s1.symbol, s1.close AS close_price, 
               (s1.close - s2.close) / s2.close AS percent_change
        FROM stock_prices s1
        LEFT JOIN stock_prices s2 
            ON s1.symbol = s2.symbol
            AND s2.date = '{previous_date}'
        WHERE s1.date = '{latest_date}'
        AND (s1.close - s2.close) / s2.close BETWEEN 0.03 AND 0.15
    """)
    rising_stocks = cursor.fetchall()

    # 매수 적합 종목 선택
    cursor.execute(f"""
    SELECT ti.symbol, ti.rsi, ti.stochastic_k, ti.macd, ti.macd_signal, 
           ti.moving_average, ti.bollinger_lower, ti.bollinger_upper, 
           ti.atr, ti.cci, sp.close AS close_price, sp.volume, 
           (SELECT AVG(volume) FROM stock_prices WHERE symbol = sp.symbol AND date >= DATE_SUB(sp.date, INTERVAL 10 DAY)) AS avg_volume
    FROM technical_indicators AS ti
    JOIN stock_prices AS sp ON ti.symbol = sp.symbol AND sp.date = ti.date
    WHERE ti.date = '{latest_date}'
      AND (
          ti.rsi < 40 OR 
          ti.stochastic_k < 25 OR 
          ti.macd > ti.macd_signal OR 
          sp.close < ti.moving_average OR 
          sp.close < ti.bollinger_lower OR 
          ti.cci < -90 OR 
          ti.atr > 0.05 * sp.close  -- ATR 값이 종가의 5% 이상인 경우 변동성 조건 추가
      )
    """)
    buy_indicators = cursor.fetchall()

    # 매수 적합 종목을 딕셔너리로 변환
    buy_indicators_dict = {item['symbol']: item for item in buy_indicators}

    # 포트폴리오 테이블에 저장
    for stock in rising_stocks:
        symbol = stock['symbol']

        # 예외 종목인 경우 건너뛰기
        if symbol in exclude_symbols:
            print(f"예외 처리된 종목: {symbol}")
            continue

        close_price = float(stock['close_price'])
        percent_change = float(stock['percent_change']) if stock['percent_change'] is not None else None

        # 매수 적합 조건을 충족하는 종목만 추가
        if percent_change is not None and symbol in buy_indicators_dict:
            indicator_data = buy_indicators_dict[symbol]
            rsi = float(indicator_data.get('rsi', 0.0))
            stochastic_k = float(indicator_data.get('stochastic_k', 0.0))
            macd = float(indicator_data.get('macd', 0.0))
            macd_signal = float(indicator_data.get('macd_signal', 0.0))
            moving_average = float(indicator_data.get('moving_average', 0.0))
            bollinger_lower = float(indicator_data.get('bollinger_lower', 0.0))
            atr = float(indicator_data.get('atr', 0.0))
            cci = float(indicator_data.get('cci', 0.0))
            volume = float(indicator_data.get('volume', 0.0))
            avg_volume = float(indicator_data.get('avg_volume', 0.0))

            # 스코어 계산
            score = 0
            if rsi < 40:
                score += 2
            if stochastic_k < 25:
                score += 2
            if macd > macd_signal:
                score += 2
            if close_price < moving_average:
                score += 1
            if volume > 1.5 * avg_volume:
                score += 2

            if score >= 4 and volume > 1.5 * avg_volume:
                selection_reason = []
                if rsi < 40:
                    selection_reason.append("RSI < 40 (과매도)")
                if stochastic_k < 25:
                    selection_reason.append("Stochastic K < 25 (과매도)")
                if macd > macd_signal:
                    selection_reason.append("MACD > Signal")
                if close_price < moving_average:
                    selection_reason.append("Close < Moving Average")
                if volume > 1.5 * avg_volume:
                    selection_reason.append("High Volume Surge")
                reason_text = "; ".join(selection_reason)

                # 포트폴리오 테이블에 삽입
                cursor.execute("""
                    INSERT INTO portfolio (symbol, date, close_price, moving_average, rsi, macd, macd_signal, 
                                           stochastic_k, stochastic_d, bollinger_upper, bollinger_lower, atr, 
                                           cci, obv, selection_reason, created_at, selected_date)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NULL, %s, NOW(), NOW())
                    ON DUPLICATE KEY UPDATE 
                        date = %s,
                        close_price = %s,
                        moving_average = %s,
                        rsi = %s,
                        macd = %s,
                        macd_signal = %s,
                        stochastic_k = %s,
                        stochastic_d = %s,
                        bollinger_upper = %s,
                        bollinger_lower = %s,
                        atr = %s,
                        cci = %s,
                        selection_reason = %s,
                        created_at = NOW(),
                        selected_date = NOW()
                """, (symbol, latest_date, close_price, moving_average, rsi, macd, macd_signal, stochastic_k, 
                      float(indicator_data.get('stochastic_d', 0.0)), float(indicator_data.get('bollinger_upper', 0.0)), 
                      bollinger_lower, atr, cci, reason_text, latest_date, close_price, moving_average, 
                      rsi, macd, macd_signal, stochastic_k, float(indicator_data.get('stochastic_d', 0.0)), 
                      float(indicator_data.get('bollinger_upper', 0.0)), bollinger_lower, atr, cci, reason_text))

    conn.commit()
    cursor.close()
    conn.close()
    print("포트폴리오 업데이트 완료")

