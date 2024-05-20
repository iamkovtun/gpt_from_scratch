from flask import Flask, request, jsonify
import torch

# Assuming the translate function and its dependencies are defined in translate.py
from translate import translate

app = Flask(__name__)

@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    try:
        translated_text = translate(text)
        return jsonify({'translated_text': translated_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
