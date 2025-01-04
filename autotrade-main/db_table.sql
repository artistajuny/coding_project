-- 1. 주식 데이터 테이블 (stock_prices)
CREATE TABLE stock_prices (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(10, 2),
    high DECIMAL(10, 2),
    low DECIMAL(10, 2),
    close DECIMAL(10, 2),
    volume BIGINT,
    UNIQUE KEY (symbol, date)
);

-- 2. 예측 데이터 테이블 (predictions)
CREATE TABLE trading_records (
    id INT AUTO_INCREMENT PRIMARY KEY,
    date DATETIME,                         -- 매매 실행 시점
    symbol VARCHAR(10),                    -- 종목 코드
    predicted_price DECIMAL(10, 2),        -- 예측 가격 (모델이 예측한 주가)
    actual_price DECIMAL(10, 2),           -- 실제 매매 시점의 주가
    action VARCHAR(10),                    -- 매매 행동 ('매수', '매도', '관망' 등)
    quantity INT,                          -- 매매 수량
    profit_loss DECIMAL(10, 2),            -- 매매에 따른 손익 (이익 또는 손실)
    profit_loss_percentage DECIMAL(5, 2),  -- 손익률 (손익을 %로 나타낸 값)
    strategy VARCHAR(50),                  -- 사용한 매매 전략명 (ex. '기술적 지표 기반', 'LSTM 예측' 등)
    notes TEXT,                            -- 매매 관련 메모 (예: 특정 조건 또는 상황)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP  -- 기록 생성 시간
);

CREATE TABLE portfolio (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    close_price DECIMAL(10, 2),
    moving_average DECIMAL(10, 2),
    rsi DECIMAL(5, 2),
    macd DECIMAL(10, 2),
    atr DECIMAL(10, 2),
    selection_reason VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY (symbol, date)
);

-- 3. 기술적 지표 테이블 (technical_indicators)
CREATE TABLE IF NOT EXISTS technical_indicators (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    moving_average DECIMAL(10, 2),        -- 이동평균
    rsi DECIMAL(5, 2),                    -- RSI
    macd DECIMAL(10, 2),                  -- MACD
    macd_signal DECIMAL(10, 2),           -- MACD 신호선
    bollinger_upper DECIMAL(10, 2),       -- 볼린저 밴드 상단
    bollinger_lower DECIMAL(10, 2),       -- 볼린저 밴드 하단
    stochastic_k DECIMAL(5, 2),           -- 스토캐스틱 %K
    stochastic_d DECIMAL(5, 2),           -- 스토캐스틱 %D
    atr DECIMAL(10, 2),                   -- 평균 진폭 (ATR)
    cci DECIMAL(10, 2),                   -- CCI
    obv BIGINT,                           -- OBV
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY (symbol, date)
);



-- 4. AI 모델 학습 데이터 테이블 (model_training_data)
CREATE TABLE model_training_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    feature1 DECIMAL(10, 2),
    feature2 DECIMAL(10, 2),
    feature3 DECIMAL(10, 2),
    target DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. 사용자 설정 테이블 (user_settings)
CREATE TABLE user_settings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    setting_name VARCHAR(50) NOT NULL,
    setting_value VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 6. 매매 기록 테이블 (trade_history)
CREATE TABLE trade_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    trade_type ENUM('buy', 'sell') NOT NULL,
    price DECIMAL(10, 2),
    volume BIGINT,
    trade_date DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 7. 로그 테이블 (logs)
CREATE TABLE logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    log_message TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 8. 주식 기본정보 테이블 (stock_info)
CREATE TABLE stock_info (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    name VARCHAR(100) NOT NULL,
    sector VARCHAR(50),
    industry VARCHAR(50),
    market_cap BIGINT,
    listing_date DATE,
    exchange VARCHAR(10),
    description TEXT
);

CREATE TABLE financial_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    current_assets DECIMAL(15, 2),        -- 유동자산
    fixed_assets DECIMAL(15, 2),          -- 고정자산
    total_assets DECIMAL(15, 2),          -- 자산총계
    current_liabilities DECIMAL(15, 2),   -- 유동부채
    fixed_liabilities DECIMAL(15, 2),     -- 고정부채
    total_liabilities DECIMAL(15, 2),     -- 부채총계
    capital_stock DECIMAL(15, 2),         -- 자본금
    capital_surplus DECIMAL(15, 2),       -- 자본 잉여금
    retained_earnings DECIMAL(15, 2),     -- 이익 잉여금
    total_equity DECIMAL(15, 2),          -- 자본총계
    revenue DECIMAL(15, 2),               -- 매출액
    cost_of_goods_sold DECIMAL(15, 2),    -- 매출 원가
    gross_profit DECIMAL(15, 2),          -- 매출 총 이익
    operating_income DECIMAL(15, 2),      -- 영업 이익
    net_income DECIMAL(15, 2),            -- 당기순이익
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY (symbol, date)
);

CREATE TABLE financial_ratios (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    
    -- 재무비율
    revenue_growth DECIMAL(10, 2),          -- 매출액 증가율
    operating_income_growth DECIMAL(10, 2), -- 영업이익 증가율
    net_income_growth DECIMAL(10, 2),       -- 순이익 증가율
    roe DECIMAL(10, 2),                     -- ROE
    eps DECIMAL(10, 2),                    -- EPS
    sps DECIMAL(10, 2),                    -- 주당매출액
    bps DECIMAL(10, 2),                    -- BPS
    reserve_ratio DECIMAL(10, 2),           -- 유보비율
    debt_ratio DECIMAL(10, 2),              -- 부채비율
    
    -- 수익성비율
    total_capital_profit_rate DECIMAL(10, 2),    -- 총자본 순이익율
    self_capital_profit_rate DECIMAL(10, 2),     -- 자기자본 순이익율
    sales_net_profit_rate DECIMAL(10, 2),        -- 매출액 순이익율
    sales_gross_profit_rate DECIMAL(10, 2),      -- 매출액 총이익율
    
    -- 기타주요비율
    payout_ratio DECIMAL(10, 2),                 -- 배당성향
    eva DECIMAL(20, 2),                         -- EVA
    ebitda DECIMAL(15, 2),                      -- EBITDA
    ev_ebitda DECIMAL(10, 2),                    -- EV_EBITDA
    
    -- 안정성비율
    borrowing_dependency DECIMAL(10, 2),         -- 차입금 의존도
    current_ratio DECIMAL(10, 2),                -- 유동비율
    quick_ratio DECIMAL(10, 2),                  -- 당좌비율
    
    -- 성장성비율
    equity_growth DECIMAL(10, 2),                -- 자기자본 증가율
    total_assets_growth DECIMAL(10, 2),          -- 총자산 증가율

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE KEY (symbol, date)
);




CREATE TABLE tick_indicators (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp DATETIME NOT NULL,
    rsi DECIMAL(5, 2) DEFAULT NULL,
    moving_average DECIMAL(10, 2) DEFAULT NULL,
    macd DECIMAL(10, 2) DEFAULT NULL,
    macd_signal DECIMAL(10, 2) DEFAULT NULL,
    bollinger_upper DECIMAL(10, 2) DEFAULT NULL,
    bollinger_lower DECIMAL(10, 2) DEFAULT NULL,
    UNIQUE KEY unique_symbol_timestamp (symbol, timestamp)
);

CREATE TABLE prediction_results (
    id INT AUTO_INCREMENT PRIMARY KEY,          -- 고유 ID
    symbol VARCHAR(10) NOT NULL,                -- 종목 코드
    prediction_date DATE NOT NULL,              -- 예측 날짜
    predicted_rise_prob FLOAT NOT NULL,         -- 상승 확률
    actual_result TINYINT,                      -- 실제 결과 (1: 상승, 0: 하락)
    is_correct TINYINT,                         -- 예측 결과가 맞았는지 여부 (1: 맞음, 0: 틀림)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- 레코드 생성 시간
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP, -- 레코드 수정 시간
    INDEX idx_symbol_date (symbol, prediction_date) -- 심볼과 날짜로 조회 속도 향상을 위한 인덱스
);
