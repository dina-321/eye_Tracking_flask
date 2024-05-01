from flask import Flask, render_template
from flask_socketio import SocketIO
from function import detect_cheating

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app)

@app.route('/')
def home():
    return render_template('index.html')

@socketio.on('get_detection_results')
def get_detection_results():
    detect_cheating(socketio)  # Pass socketio to detect_cheating function

if __name__ == "__main__":
    socketio.run(app, debug=True)
