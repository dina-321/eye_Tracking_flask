from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from function import detect_cheating

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app)

@app.route('/')
def home():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('get_detection_results')
def get_detection_results():
    detection_results = detect_cheating()
    
    # Convert Timestamp objects to strings
    detection_results['Time'] = detection_results['Time'].astype(str)
    
    # Emit the modified DataFrame as a list of dictionaries
    emit('update_detection_results', detection_results.to_dict(orient='records'))


if __name__ == "__main__":
    socketio.run(app, debug=True)
