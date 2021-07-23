# Run by typing python3 main.py

## **IMPORTANT:** only collaborators on the project where you run
## this can access this web server!


# import basics
import os
from demo_cli import main
import datetime

# import stuff for our web server
from flask import Flask, flash, request, redirect, url_for, render_template
from flask import send_from_directory
from flask import jsonify
from utils_main import get_base_url, allowed_file, and_syntax

# import stuff for our models
import torch
from aitextgen import aitextgen
import re
import json


# from synthesizer.inference import Synthesizer
# from encoder import inference as encoder
# from vocoder import inference as vocoder
# from pathlib import Path
# import numpy as np
# import librosa
# encoder_weights = Path("pretrained/encoder/saved_models/pretrained.pt")
# vocoder_weights = Path("pretrained/vocoder/saved_models/pretrained/pretrained.pt")
# syn_dir = Path("pretrained/synthesizer/saved_models/logs-pretrained/taco_pretrained")
# encoder.load_model(encoder_weights)
# synthesizer = Synthesizer(syn_dir)
# vocoder.load_model(vocoder_weights)
'''
Coding center code - comment out the following 4 lines of code when ready for production
'''
# load up the model into memory
# you will need to have all your trained model in the app/ directory.
ai_Tupac = aitextgen(to_gpu=False, model_folder='trained_model/Tupac_Model')
ai_Eminem = aitextgen(to_gpu=False, model_folder='trained_model/Eminem_Model')
ai_TTC = aitextgen(to_gpu=False, model_folder='trained_model/TtC_Model')


# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
#port = 12347
#base_url = get_base_url(port)
#app = Flask(__name__, static_url_path=base_url+'static')

'''
Deployment code - uncomment the following line of code when ready for production
'''
app = Flask(__name__)

@app.route('/')
#@app.route(base_url)
def home():
    return render_template('Overview.html', generated=None)

# 
#     return render_template('Overview.html#sec-1fe0', generated=None)

# @app.route(base_url)
# def About():
#     return render_template('Overview.html#sec-2750', generated=None)

# @app.route(base_url)
# def Tools():
#     return render_template('Overview.html#sec-0a7f', generated=None)

# @app.route(base_url)
# def Try():
#     return render_template('Overview.html#sec-896a', generated=None)

@app.route('/Tupac')
def Tupac_function():
    return render_template('Tupac_page.html', generated=None)

@app.route('/Eminem')
def Eminem_function():
    return render_template('Eminem_page.html', generated=None)

@app.route('/Tyler')
def TTC_function():
    return render_template('TTC_page.html', generated=None)

@app.route('/', methods=['POST'])
#@app.route(base_url, methods=['POST'])
def home_post():
    return redirect(url_for('results'))

@app.route('/results')
#@app.route(base_url + '/results')
def results():
    return render_template('Tupac_page.html', generated=None)

# #--------------Eminem--------------------
@app.route('/generate_text_eminem', methods=["POST"])
def generate_text_eminem():
    """
    view function that will return json response for generated text. 
    """
    prompt = request.form['prompt']
    if prompt is not None:
        generated = ai_Eminem.generate(
            n=3,
            batch_size=3,
            prompt=str(prompt),
            max_length=50,
            temperature=0.9,
            return_as_list=True
        )

    formated_data = []
    for line in generated:
        formated_data.append(re.sub(r"\\n","<br>",line))

    data = {'generated_ls': formated_data}

    return jsonify(data)

#--------------Tupac--------------------

@app.route('/generate_text', methods=["POST"])
def generate_text():
    """
    view function that will return json response for generated text. 
    """

    prompt = request.form['prompt']
    print("****************************************")
    print(prompt)
    print("****************************************")
    if prompt is not None:
        generated = ai_Tupac.generate(
            n=3,
            batch_size=3,
            prompt=str(prompt),
            max_length=50,
            temperature=0.9,
            return_as_list=True
        )

    formated_data = []
    for line in generated:
        formated_data.append(re.sub(r"\\n","<br>",line))

    data = {'generated_ls': formated_data}

    return jsonify(data)

# Generate clone voice
# @app.route(base_url + '/clone', methods=["POST"])
# def clone():
#     """
#     view function that will return json response for generated text. 
#     """
    
#     prompt = request.form['prompt']
#     print("****************************************")
#     print(prompt)
#     print("****************************************")
#     if prompt is not None:
#         main("./trump11.wav",prompt)

#     data = {'generated_ls': ["Audio File Already Generated, press the button to play!!!", "Let's hear Eminem"]}
    
#     return jsonify(data)

@app.route('/clone', methods=["POST"])
def clone():
    """
    view function that will return json response for generated text. 
    """

    prompt = request.form['prompt']
    print("****************************************")
    print(prompt)
    print("****************************************")
#     current_time = str(datetime.datetime.now())
    if prompt is not None:
        
        main('./trump11.wav',prompt)
#         generated = ai_Tupac.generate(
#             n=3,
#             batch_size=3,
#             prompt=str(prompt),
#             max_length=50,
#             temperature=0.9,
#             return_as_list=True
#         )

#     formated_data = []
#     for line in generated:
#         formated_data.append(re.sub(r"\\n","<br>",line))

#     data = {'generated_ls': formated_data}

#     return jsonify(data)


# #--------------TTC--------------------
@app.route('/generate_text_ttc', methods=["POST"])
def generate_text_ttc():
    """
    view function that will return json response for generated text. 
    """

    prompt = request.form['prompt']
    if prompt is not None:
        generated = ai_TTC.generate(
            n=3,
            batch_size=3,
            prompt=str(prompt),
            max_length=50,
            temperature=0.9,
            return_as_list=True
        )

    formated_data = []
    for line in generated:
        formated_data.append(re.sub(r"\\n","<br>",line))

    data = {'generated_ls': formated_data}

    return jsonify(data)

if __name__ == "__main__":
    '''
    coding center code
    '''
    # IMPORTANT: change the cocalcx.ai-camp.org to the site where you are editing this file.
    website_url = 'cocalc1.ai-camp.org'
    print(f"Try to open\n\n    https://{website_url}" + base_url + '\n\n')

    app.run(host = '0.0.0.0', port=port, debug=True)
    import sys; sys.exit(0)

    '''
    scaffold code
    '''
    # Only for debugging while developing
    # app.run(port=80, debug=True)
