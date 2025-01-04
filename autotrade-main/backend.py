from flask import Flask, request, jsonify
import bcrypt
from db_utils import get_db_connection
from flask_cors import CORS
from cryptography.fernet import Fernet

app = Flask(__name__)
CORS(app)

# 암호화를 위한 대칭키 생성
encryption_key = Fernet.generate_key()
cipher_suite = Fernet(encryption_key)

# 중복 확인 API
@app.route('/check-username', methods=['POST'])
def check_username():
    data = request.get_json()
    username = data.get('username')

    if not username:
        return jsonify({'error': '아이디를 입력해주세요.'}), 400

    conn = get_db_connection()
    cursor = conn.cursor()

    # 데이터베이스에서 해당 아이디가 있는지 확인
    sql = "SELECT COUNT(*) FROM users WHERE username = %s"
    cursor.execute(sql, (username,))
    result = cursor.fetchone()

    cursor.close()
    conn.close()

    if result[0] > 0:
        return jsonify({'available': False}), 200
    else:
        return jsonify({'available': True}), 200

# 회원가입 API
@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()

    username = data.get('username')
    password = data.get('password')
    account_number = data.get('accountNumber')
    app_key = data.get('appKey')
    app_secret = data.get('appSecret')

    if not all([username, password, account_number, app_key, app_secret]):
        return jsonify({'error': '모든 필드를 입력해주세요.'}), 400

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    encrypted_app_key = cipher_suite.encrypt(app_key.encode('utf-8'))
    encrypted_app_secret = cipher_suite.encrypt(app_secret.encode('utf-8'))

    conn = get_db_connection()
    cursor = conn.cursor()

    sql = """
    INSERT INTO users (username, password, account_number, app_key, app_secret) 
    VALUES (%s, %s, %s, %s, %s)
    """
    try:
        cursor.execute(sql, (username, hashed_password, account_number, encrypted_app_key, encrypted_app_secret))
        conn.commit()
        return jsonify({'message': '회원가입이 성공적으로 완료되었습니다!'}), 201
    except Exception as e:
        conn.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)