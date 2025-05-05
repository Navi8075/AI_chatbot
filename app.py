# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 09:03:39 2025

@author: z048540
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from flask import Flask, request, jsonify,send_file
import llm_chatbot_script  # Assuming you have your LLM code in this script
import io
from gtts import gTTS
#from pydub import AudioSegment

app = Flask(__name__)

@app.route('/test')
def hello():
    print('request recieved')
    return 'Hello'

@app.route('/chat', methods=['POST'])
def chat():
    print("request recieved")
    if 'audio' not in request.files:
        print(" No file part in request.")
        return jsonify({'error': 'No file part'}), 400 
    query = ''
    audio = request.files['audio']
    audio_byte = audio.read()
    audio_file = io.BytesIO(audio_byte)
    
    
    query,response = llm_chatbot_script.chatbot_response(audio_file)
    #return jsonify({"query":query,"response": response}), 200

    tts = gTTS(response)
    mp3_io = io.BytesIO()
    tts.write_to_fp(mp3_io)
    mp3_io.seek(0)

    # Step 2: Convert MP3 to WAV using pydub
    '''audio = AudioSegment.from_file(mp3_io, format="mp3")
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)
    '''
    # Step 3: Return WAV audio
    return send_file(mp3_io, mimetype='audio/mpeg', as_attachment=False, download_name="speech.mp3")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)  