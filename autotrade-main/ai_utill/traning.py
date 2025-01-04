import numpy as np
import mysql.connector
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import yaml

# 환경 초기 설정
def get_db_connection():
    """MySQL 데이터베이스 연결 생성"""
    with open('D:\\coding_project\\autotrade-main\\config.yaml', encoding='UTF-8') as f:
        _cfg = yaml.load(f, Loader=yaml.FullLoader)

    conn = mysql.connector.connect(
        host=_cfg['DB_HOST'],
        user=_cfg['DB_USER'],
        password=_cfg['DB_PASS'],
        database=_cfg['DB_NAME']
    )
    return conn

# 학습 데이터 준비 함수
def prepare_training_data():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT symbol, predicted_price, actual_price, action, profit_loss_percentage FROM trading_records")
    rows = cursor.fetchall()
    conn.close()

    X, y, rewards = [], [], []
    action_map = {'매수': 0, '매도': 1, '관망': 2}

    for row in rows:
        symbol, predicted_price, actual_price, action, profit_loss_percentage = row
        feature_vector = [float(predicted_price), float(actual_price)]
        X.append(feature_vector)
        y.append(action_map[action])
        rewards.append(float(profit_loss_percentage))

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return np.array(X_scaled), np.array(y), np.array(rewards)

# 상태-행동 가치 함수 (Q-네트워크) 구축
def build_q_network(input_shape):
    model = Sequential()
    model.add(Dense(64, input_dim=input_shape, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='linear'))  # 행동 3가지: 매수, 매도, 관망
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# 강화학습 알고리즘 (DQN)
def train_dqn(episodes=100):
    X, y, rewards = prepare_training_data()
    n_actions = 3
    q_network = build_q_network(X.shape[1])

    epsilon = 1.0  # 탐색 비율 (탐험/탐사 조절)
    epsilon_min = 0.01
    epsilon_decay = 0.995
    gamma = 0.95  # 할인율

    for episode in range(episodes):
        total_reward = 0
        for i in range(len(X)):
            state = np.reshape(X[i], [1, X.shape[1]])

            # Epsilon-greedy 정책에 따른 행동 선택
            if np.random.rand() <= epsilon:
                action = np.random.choice(n_actions)
            else:
                q_values = q_network.predict(state)
                action = np.argmax(q_values[0])

            # 보상 및 다음 상태
            reward = rewards[i]
            next_state = state if i == len(X) - 1 else np.reshape(X[i + 1], [1, X.shape[1]])
            total_reward += reward

            # Q 값 업데이트
            target = reward
            if i < len(X) - 1:
                next_q_values = q_network.predict(next_state)
                target += gamma * np.amax(next_q_values[0])

            target_q_values = q_network.predict(state)
            target_q_values[0][action] = target

            # Q 네트워크 학습
            q_network.fit(state, target_q_values, epochs=1, verbose=0)

        # Epsilon 감소
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"에피소드 {episode + 1}/{episodes}, 총 보상: {total_reward}")

    q_network.save("dqn_trading_model.h5")
    print("강화학습 모델이 'dqn_trading_model.h5'로 저장되었습니다.")

# 학습 실행
train_dqn(episodes=100)
