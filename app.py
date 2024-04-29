from flask import Flask, render_template
from function import detect_cheating

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/update_results')
def update_results():
    detection_results = detect_cheating()  # Call detect_cheating to get the latest results
    return detection_results.to_json()  # Convert the DataFrame to JSON and return it

if __name__ == "__main__":
    app.run(debug=True)