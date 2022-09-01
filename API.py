from flask import Flask, request, jsonify
from flask_cors import CORS

import yaml, os, argparse, base64, re, uuid
from scipy.io import wavfile

from Inference import Inferencer
from Arg_Parser import Recursive_Parse

from Datasets import Text_to_Token

regex_checker = re.compile('[가-힣A-Z,.?!\'\-\s]+')

# flask app
app = Flask(__name__, static_url_path='/static')
CORS(app)

@app.route('/tts', methods=['POST'])
def tts():
    text = request.json['text'].strip()
    speaker = request.json['speaker'].strip()
    emotion = request.json['emotion'].strip() if 'emotion' in request.json.keys() else 'Neutral'
    length_scale = 1.0 / float(request.json['length_scale'] if 'length_scale' in request.json.keys() else 1.0)
    
    audio = inferencer.Inference_Epoch(
        texts= [text],
        speakers= [speaker],
        emotions= [emotion],
        length_scales= length_scale
        )[1][0]

    path = f'{uuid.uuid4()}.wav'
    wavfile.write(
        filename= path,
        data= audio,
        rate= hp.Sound.Sample_Rate,
        )
    with open(path, 'rb') as f:
        encodings = base64.b64encode(f.read())
    payload = {'audio': encodings.decode()}

    os.remove(path)

    return jsonify(payload)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-hp', '--hyper_parameters', required= True, type= str)
    parser.add_argument('-path', '--checkpoint_path', type= str)
    parser.add_argument('-port', '--port', type= int)

    args = parser.parse_args()

    global hp, inferencer

    hp = Recursive_Parse(yaml.load(
        open(args.hyper_parameters, encoding='utf-8'),
        Loader=yaml.Loader
        ))
    
    inferencer = Inferencer(
        hp_path= args.hyper_parameters,
        checkpoint_path= args.checkpoint_path,
        batch_size= 1
        )
        
    app.run(host='0.0.0.0', port=args.port, debug=False)

# CUDA_VISIBLE_DEVICES=0 python API.py -hp Hyper_Parameters.yaml -path "C:/Users/Heejo.You/Downloads/Telegram Desktop/S_25000.pt" -port 8001