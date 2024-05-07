
from flask import Flask, request, jsonify
from function import detect_cheating
import tempfile
import os

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect():
    if request.method == 'POST':
        if 'image_files' not in request.files:
            return jsonify({'error': 'No files uploaded'})
        
        image_files = request.files.getlist('image_files')
        if len(image_files) == 0:
            return jsonify({'error': 'No files uploaded'})
        
        # Save uploaded files temporarily
        temp_file_paths = []
        for image_file in image_files:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_file_paths.append(temp_file.name)
            image_file.save(temp_file)
            temp_file.close()
        
        # Perform cheating detection
        results = detect_cheating(temp_file_paths)
        
        # Delete temporary files
        for temp_file_path in temp_file_paths:
            os.unlink(temp_file_path)
        
        return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
