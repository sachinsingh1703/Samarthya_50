from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_socketio import SocketIO, emit
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
socketio = SocketIO(app)

# Load the trained model and scaler
model = load_model('model_ddos.h5')
scaler = joblib.load('scaler.save')

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
        user = User(username=username, password=password)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Login unsuccessful. Please check your credentials', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    if request.method == 'POST':
        try:
            website_link = request.form['website_link']
            features = extract_features(website_link)
            features = np.reshape(features, (1, -1))
            features = scaler.transform(features)
            prediction = model.predict(features)
            is_ddos_attack = prediction[0][0] > 0.5
            return render_template('result.html', website_link=website_link, is_ddos_attack=is_ddos_attack)
        except Exception as e:
            return render_template('index.html', error=str(e))
    return render_template('index.html')

def extract_features(website_link):
    url_length = len(website_link)
    return np.array([[url_length] * 10])

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('monitor_website')
def handle_monitor_website(data):
    website_link = data['website_link']
    # Simulate real-time data for demo purposes
    packets_received = np.random.randint(0, 100)
    emit('update_graph', {'packets_received': packets_received}, broadcast=True)

if __name__ == '__main__':
    db.create_all()
    socketio.run(app, debug=True)
