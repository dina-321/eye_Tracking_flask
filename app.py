from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from function import detect_cheating

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
socketio = SocketIO(app)

@app.route('/detect_cheating', methods=['POST'])
def detect_cheating_route():
    detection_results_df = detect_cheating(socketio)  # Trigger detection process
    detection_results_dict = detection_results_df.to_dict(orient='records')  # Convert DataFrame to dictionary
    return jsonify(detection_results_dict)  # Convert dictionary to JSON and return as response

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('get_detection_results')
def get_detection_results():
    detect_cheating(socketio)  # Pass socketio to detect_cheating function

if __name__ == "__main__":
    socketio.run(app, debug=True)

