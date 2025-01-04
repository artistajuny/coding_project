import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import yaml
from datetime import datetime
from collections import deque
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # TensorFlow를 CPU 모드에서 실행
# 1. 데이터베이스 연결 설정
with open('D:\\coding_project\\autotrade-main\\config.yaml', encoding='UTF-8') as f:
    _cfg = yaml.load(f, Loader=yaml.FullLoader)

def get_db_connection():
    db_password = _cfg['DB_PASS'].replace("@", "%40").replace("#", "%23")
    db_url = f"mysql+pymysql://{_cfg['DB_USER']}:{db_password}@{_cfg['DB_HOST']}/{_cfg['DB_NAME']}"
    engine = create_engine(db_url)
    return engine

# 데이터 매치 및 전처리
def fetch_data(engine, symbol):
    """심볼에 대한 데이터 조회 및 병합"""
    query = f"""
    SELECT sd.symbol, CONCAT(CURDATE(), ' ', sd.trade_time) AS timestamp, sd.price, 
           ti.rsi, ti.moving_average, ti.macd, ti.macd_signal, ti.bollinger_upper, ti.bollinger_lower
    FROM stock_tick_data sd
    LEFT JOIN tick_indicators ti ON sd.symbol = ti.symbol AND CONCAT(CURDATE(), ' ', sd.trade_time) = ti.timestamp
    WHERE sd.symbol = '{symbol}' AND sd.is_active = 1
    ORDER BY sd.timestamp ASC
    """
    df = pd.read_sql(query, engine)

    # 'timestamp' 열 제거 후, 나머지 열을 수치형 데이터로 스케일링
    scaler = MinMaxScaler()
    numeric_columns = ['price', 'rsi', 'moving_average', 'macd', 'macd_signal', 'bollinger_upper', 'bollinger_lower']
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns].fillna(method='ffill').fillna(method='bfill'))

    return df[numeric_columns], scaler

# 2. DQN 모델 설계
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # 할인 요인
        self.epsilon = 1.0  # 탐색률
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(20, self.state_size)))  # 시계열 길이 20, 특성 개수로 설정
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state.astype('float32'))[0]))
            target_f = self.model.predict(state.astype('float32'))
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 4. 환경 상호작용 및 학습 루프
def train_dqn_agent(agent, data, episodes=100):
    state_size = data.shape[1]
    action_size = 2  # 상승, 하락 예측을 위한 두 가지 행동
    batch_size = 16

    for e in range(episodes):
        # 초기 상태 설정 (20일치 데이터를 사용)
        for time in range(20, len(data) - 1):
            state = data.iloc[time - 20:time].values.reshape(1, 20, state_size).astype('float32')
            action = agent.act(state)
            next_state = data.iloc[time - 19:time + 1].values.reshape(1, 20, state_size).astype('float32')
            reward = 1 if (data.iloc[time + 1]['price'] > data.iloc[time]['price'] and action == 1) else -1
            done = time == len(data) - 21
            agent.remember(state, action, reward, next_state, done)
            if done:
                print(f"episode: {e+1}/{episodes}, score: {time}, epsilon: {agent.epsilon:.2}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

# 5. 모든 심볼에 대해 모델 학습
def train_all_symbols(engine):
    # stock_tick_data 테이블에서 고유한 심볼 목록 가져오기
    symbols = pd.read_sql("SELECT DISTINCT symbol FROM stock_tick_data WHERE is_active = 1", engine)['symbol'].tolist()

    for symbol in symbols:
        data, scaler = fetch_data(engine, symbol)

        if not data.empty:
            state_size = data.shape[1]
            action_size = 2
            agent = DQNAgent(state_size, action_size)
            train_dqn_agent(agent, data)
            print(f"{symbol}에 대한 DQN 모델 학습 완료")

            # 모델 저장
            os.makedirs("models", exist_ok=True)
            agent.model.save(f"models/{symbol}_dqn_model")
            print(f"{symbol}에 대한 DQN 모델이 저장되었습니다.")

# 실행 코드
if __name__ == "__main__":
    engine = get_db_connection()
    train_all_symbols(engine)
