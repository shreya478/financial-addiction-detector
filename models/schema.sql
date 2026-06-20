CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    archetype VARCHAR(20) NOT NULL, -- 'casual', 'frequent', 'compulsive'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE sessions (
    session_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    device_type VARCHAR(50)
);

CREATE TABLE transactions (
    transaction_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    session_id INTEGER REFERENCES sessions(session_id),
    timestamp TIMESTAMP NOT NULL,
    amount DECIMAL(10, 2) NOT NULL,
    transaction_type VARCHAR(20) NOT NULL, -- 'deposit', 'withdrawal', 'bet', 'win', 'loss', 'transfer_in'
    payment_method VARCHAR(50),
    external_transfer BOOLEAN DEFAULT FALSE -- Flag for bailout-seeking
);

CREATE TABLE labels (
    label_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    preoccupation_score INTEGER DEFAULT 0,
    tolerance_score INTEGER DEFAULT 0,
    withdrawal_score INTEGER DEFAULT 0,
    loss_of_control_score INTEGER DEFAULT 0,
    loss_chasing_score INTEGER DEFAULT 0,
    lying_score INTEGER DEFAULT 0,
    escapism_score INTEGER DEFAULT 0,
    jeopardizing_score INTEGER DEFAULT 0,
    bailout_score INTEGER DEFAULT 0,
    total_score INTEGER DEFAULT 0,
    risk_tier VARCHAR(20), -- 'Low', 'Moderate', 'High'
    labeled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
