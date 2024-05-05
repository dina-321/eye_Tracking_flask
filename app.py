from flask import Flask, request, jsonify
from function import detect_cheating

app = Flask(__name__)

# Sample database connection
# db = connect_to_database()

@app.route('/detect_cheating', methods=['POST'])
def detect_cheating_route():
    try:
        # Get the video URL from the request
        video_url = request.form['video_url']
        
        # Call the detect_cheating function with the video URL
        results = detect_cheating(video_url)
        
        # Return the results as JSON
        return jsonify(results)
    except Exception as e:
        # Handle exceptions gracefully
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({'error': 'An error occurred'}), 500

if __name__ == "__main__":
    # Initialize any resources here (e.g., database connections)
    # db.connect()
    
    # Run the application using Gunicorn
    app.run(debug=True)
