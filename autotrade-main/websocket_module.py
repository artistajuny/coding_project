import websocket
import json
import requests
import yaml
from collections import deque
from ai_model import analyze_signals_with_trade_data
import time

# 설정 파일 로드
with open('D:\\coding_project\\autotrade-main\\config.yaml', encoding='UTF-8') as f:
    _cfg = yaml.load(f, Loader=yaml.FullLoader)

APP_KEY = _cfg['APP_KEY']
APP_SECRET = _cfg['APP_SECRET']
URL_BASE = _cfg['URL_BASE']
WEBSOCKET_APPROVAL_KEY = ""

# 체결 데이터를 저장할 딕셔너리
close_data_dict = {}

# AES256 복호화 함수 (필요 시 사용할 수 있음)
def decrypt_data(key, iv, cipher_text):
    from Crypto.Cipher import AES
    from Crypto.Util.Padding import unpad
    from base64 import b64decode

    cipher = AES.new(key.encode('utf-8'), AES.MODE_CBC, iv.encode('utf-8'))
    decrypted = unpad(cipher.decrypt(b64decode(cipher_text)), AES.block_size)
    return decrypted.decode('utf-8')

# 웹소켓 승인 키 발급 함수
def get_websocket_approval_key(retry_count=3):
    """웹소켓 접속키 발급"""
    global WEBSOCKET_APPROVAL_KEY
    headers = {"content-type": "application/json; charset=utf-8"}
    body = {
        "grant_type": "client_credentials",
        "appkey": APP_KEY,
        "secretkey": APP_SECRET
    }
    URL = f"{URL_BASE}/oauth2/Approval"

    for _ in range(retry_count):
        try:
            res = requests.post(URL, headers=headers, data=json.dumps(body))
            res.raise_for_status()
            WEBSOCKET_APPROVAL_KEY = res.json().get('approval_key')
            if WEBSOCKET_APPROVAL_KEY:
                print("웹소켓 접속키 발급 성공")
                return
            else:
                print("웹소켓 접속키 발급 실패.")
        except requests.exceptions.RequestException as e:
            print(f"웹소켓 접속키 발급 실패: {e}")
        time.sleep(1)
    print("웹소켓 접속키 발급에 최종 실패했습니다.")

# 웹소켓 메시지 처리 함수
def on_message(ws, message, callback):
    """웹소켓 메시지 처리"""
    global close_data_dict

    if "PINGPONG" in message:  # Ping-Pong 메시지 처리
        ws.send("PINGPONG", websocket.ABNF.OPCODE_PONG)  # Pong 메시지 응답
        print("Ping-Pong 메시지 처리 완료")
        return

    try:
        data_list = message.split('|')

        # 실시간 체결 데이터만 처리
        if data_list[0] == "0" and data_list[1] == "H0STCNT0":
            raw_data = data_list[3:]  # 체결 데이터

            for data_block in raw_data:
                stock_data_parts = data_block.split('^')
                if len(stock_data_parts) < 30:
                    continue

                stock_code = stock_data_parts[0]  # 종목 코드
                trade_price = int(stock_data_parts[2]) if stock_data_parts[2].isdigit() else None
                trade_volume = int(stock_data_parts[12]) if stock_data_parts[12].isdigit() else None
                trade_strength = float(stock_data_parts[18]) if stock_data_parts[18].replace('.', '', 1).isdigit() else None
                change_rate = float(stock_data_parts[5]) if stock_data_parts[5].replace('.', '', 1).isdigit() else None

                # 단주 거래량 10 미만은 제외
                if trade_volume is not None and trade_volume < 10:
                    continue

                # 종목별 데이터를 close_data_dict에 추가
                if stock_code not in close_data_dict:
                    close_data_dict[stock_code] = deque(maxlen=30)

                close_data_dict[stock_code].append({
                    'price': trade_price,
                    'trade_volume': trade_volume,
                    'trade_strength': trade_strength,
                    'change_rate': change_rate,
                })

                # AI 모델에 넘길 데이터가 충분히 모였는지 확인
                if len(close_data_dict[stock_code]) >= 30:
                    data_to_pass = list(close_data_dict[stock_code])

                    # AI 모델을 호출하여 결과 예측
                    try:
                        ai_result = analyze_signals_with_trade_data(stock_code, data_to_pass)
                        predicted_action = ai_result.get('predicted_action')  # AI 예측 결과
                        indicators = ai_result.get('indicators')  # AI에서 계산된 지표
                    except Exception as e:
                        print(f"AI 모델 호출 실패: {e}")
                        predicted_action = None
                        indicators = None

                    # 콜백 함수 호출
                    if callback:
                        trade_data = {
                            'symbol': stock_code,
                            'current_price': trade_price,
                            'trade_volume': trade_volume,
                            'trade_strength': trade_strength,
                            'change_rate': change_rate,
                            'predicted_action': predicted_action,
                            'indicators': indicators
                        }
                        callback(trade_data, ws)  # ws 인자 포함
    except Exception as e:
        print(f"메시지 처리 중 오류: {e} - 전체 메시지: {message}")

# 웹소켓 구독 업데이트 함수
def update_websocket_subscription(ws, symbol, action):
    """웹소켓 구독 추가/제거"""
    if action == "add":
        subscribe_message = {
            "header": {
                "approval_key": WEBSOCKET_APPROVAL_KEY,
                "custtype": "P",
                "tr_type": "1",
                "content-type": "utf-8"
            },
            "body": {
                "input": {
                    "tr_id": "H0STCNT0",
                    "tr_key": symbol
                }
            }
        }
        ws.send(json.dumps(subscribe_message))
        print(f"웹소켓 구독 추가: {symbol}")
    elif action == "remove":
        unsubscribe_message = {
            "header": {
                "approval_key": WEBSOCKET_APPROVAL_KEY,
                "custtype": "P",
                "tr_type": "1",
                "content-type": "utf-8"
            },
            "body": {
                "input": {
                    "tr_id": "H0STCNT0",
                    "tr_key": symbol,
                    "action": "unsubscribe"  # 가정: 'unsubscribe' 액션 지원
                }
            }
        }
        ws.send(json.dumps(unsubscribe_message))
        print(f"웹소켓 구독 제거: {symbol}")

# 웹소켓 에러 및 종료 콜백 함수
def on_error(ws, error):
    print(f"[DEBUG] 웹소켓 에러: {error}")

def on_close(ws, close_status_code, close_msg):
    """웹소켓 연결 종료 콜백"""
    print(f"웹소켓 연결이 닫혔습니다. 코드: {close_status_code}, 메시지: {close_msg}")
    print("5초 후 웹소켓 재연결을 시도합니다.")
    time.sleep(5)  # 재연결 대기

# 웹소켓 연결 및 구독 요청 함수
def on_open(ws, symbols):
    for symbol in symbols:
        update_websocket_subscription(ws, symbol, "add")
    print(f"Subscribed to real-time data for symbols: {symbols}")

def start_websocket(symbols, callback=None, max_retries=5, initial_delay=5, max_delay=60):
    """웹소켓 연결 및 데이터 처리 (지수 백오프 적용)"""
    retry_count = 0
    delay = initial_delay

    def connect():
        if not WEBSOCKET_APPROVAL_KEY:
            get_websocket_approval_key()

        if WEBSOCKET_APPROVAL_KEY:
            websocket_url = "ws://ops.koreainvestment.com:21000"
            ws = websocket.WebSocketApp(
                websocket_url,
                on_open=lambda ws: on_open(ws, symbols),
                on_message=lambda ws, message: on_message(ws, message, callback),
                on_error=on_error,
                on_close=on_close
            )
            ws.run_forever(ping_interval=30, ping_timeout=10)
            return ws
        else:
            print("웹소켓 접속키 발급에 실패하여 연결을 시도할 수 없습니다.")
            return None

    while retry_count < max_retries:
        ws = connect()
        if ws:
            return ws  # 성공적으로 연결된 경우 함수 종료
        retry_count += 1
        print(f"웹소켓 연결 실패. {delay}초 후 재시도합니다. (시도 횟수: {retry_count}/{max_retries})")
        time.sleep(delay)
        delay = min(max_delay, delay * 2)  # 지수 백오프 적용, 최대 딜레이 제한

    print("최대 재연결 시도 횟수를 초과했습니다. 웹소켓을 종료합니다.")
    return None
