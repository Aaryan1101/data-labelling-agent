#!/usr/bin/env python3
"""
Simple Flask server to serve the interface and proxy requests to ZyndAI agent.
"""

from flask import Flask, request, jsonify, send_file
import os
import requests
import json
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ZyndAI webhook URL
ZYND_WEBHOOK_URL = "http://localhost:5003/webhook"

@app.route('/')
def index():
    return send_file('interface.html')

@app.route('/process', methods=['POST'])
def process():
    """Process file and command, send to ZyndAI agent"""
    try:
        # Check for file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        command = request.form.get('command', '')
        
        if not command:
            return jsonify({'error': 'No command provided'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join('temp', filename)
        file.save(filepath)
        
        # Prepare request to ZyndAI
        files = {'file': open(filepath, 'rb')}
        data = {
            'content': f'{command} with {filename}',
            'chat_history': []
        }
        
        # Send to ZyndAI agent
        response = requests.post(ZYND_WEBHOOK_URL, files=files, data=data)
        
        # Clean up temp file
        os.remove(filepath)
        
        return jsonify(response.json()), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs('temp', exist_ok=True)
    app.run(host='0.0.0.0', port=8080, debug=True)
