from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import json
import librosa
import asyncio
import warnings
import numpy as np
import pandas as pd
from pydub import AudioSegment
import soundfile as sf
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from keras.models import load_model
plt.switch_backend('agg')

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", UserWarning) 

app = Flask(__name__)

# model=pickle.load(open('model.pkl','rb'))
model = load_model('SER.h5')

def get_file_extension(file_path):
    _, file_extension = os.path.splitext(file_path)
    return file_extension.lower()

def resize_array(array):
    new_matrix = np.zeros((30,150))   # Initialize the new matrix shape with an array 30X150 of zeros
    for i in range(30):               # Iterate rows
        for j in range(150):          # Iterate columns
            try:                                 # the mfccs of a sample will replace the matrix of zeros, then cutting the array up to 150
                new_matrix[i][j] = array[i][j]
            except IndexError:                   # if mfccs of a sample is shorter than 150, then keep looping to extend lenght to 150 with 0s
                pass
    return new_matrix

def preprocessing(audio):
    data, sampling_rate = librosa.load(audio,sr=16000)
    sampling_rate = np.array(sampling_rate)
    mfccs = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=30)
    # print("MFCCS : ",mfccs)
    r_mfccs=resize_array(mfccs)
    m_arr = np.array([r_mfccs])
    with open('mean_std_values.json', 'r') as file:
        saved_data = json.load(file)
        t_mean = np.array(saved_data['mean'])
        t_std = np.array(saved_data['std'])
    m_arr=(m_arr - t_mean)/t_std
    m_arr = m_arr[..., None]
    # print(m_arr)
    return m_arr

def waveform(audio):
    data, sampling_rate = librosa.load(audio,sr=16000)
    # plt.figure(figsize=(15, 5))
    librosa.display.waveshow(data, sr=sampling_rate)
    plt.savefig('waveform.png')
    plt.close()

def mfcc_img(audio):
    data, sampling_rate = librosa.load(audio,sr=16000)
    # Extract 'n_mfcc' numbers of MFCCs components - in this case 20
    x_mfccs = librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=20)
    # Plot MFCCs
    librosa.display.specshow(x_mfccs, sr=sampling_rate, x_axis="time", norm=Normalize(vmin=-50, vmax=50))
    plt.savefig('mfcc.png')
    plt.close()

def pychart_img(pred_arr):
    y=pred_arr[0]
    mylabels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise","calm"]
    plt.pie(y, labels = mylabels)
    plt.title("Predictions")
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig('pychart.png')
    plt.close()

def res(pred_arr):
    print(pred_arr)
    livePred=pred_arr.argmax(axis=1)
    print(livePred)
    liveabc=livePred.astype(int).flatten()
    df = pd.DataFrame(liveabc)
    print("df : ",df)
    df.replace({0:'angry',1:'disgust',2:'fear', 3:'happy', 4:'neutral', 5:'sad',6:'surprise',7: 'calm'}, inplace=True)
    r=df.iloc[0,0]
    
    print("R ", r )
    return r

@app.route('/')
def index():
    return render_template("login.html")

@app.route('/home')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form["username"]
        password = request.form["password"]
        if username == "U101" and password == "123":
            # return 'Login successful!'
            return redirect(url_for("home"))
    return render_template('login.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if os.path.exists("uploaded_audio.wav"):
        os.remove("uploaded_audio.wav")
    if os.path.exists("waveform.png"):
        os.remove("waveform.png")
    if os.path.exists("mfcc.png"):
        os.remove("mfcc.png")
    if os.path.exists("pychart.png"):
        os.remove("pychart.png")
    if "fileup" not in request.files:
        return "No file uploaded"
    fileup = request.files["fileup"]
    ext = get_file_extension(fileup.filename)
    if ext=='.mp3':
        fileup.save('uploaded_audio.mp3')
        # assign files
        input_file = "uploaded_audio.mp3"
        output_file = "uploaded_audio.wav"
        
        # convert mp3 file to wav file
        sound = AudioSegment.from_mp3(input_file)
        sound.export(output_file, format="wav")
        os.remove(input_file)
    elif ext=='.wav':
        fileup.save('uploaded_audio.wav')
    else:
        return "Please Enter file"
    
    if fileup.filename == '':
        return "No file selected"
    file_names = list(request.files.keys())
    print(file_names)
    print("File uploaded")
    waveform('uploaded_audio.wav')
    mfcc_img('uploaded_audio.wav')
    m_arr = preprocessing('uploaded_audio.wav')
    print("preprocessing done")
    prediction=model.predict(m_arr)
    pychart_img(prediction)
    print("Predict done")
    result=res(prediction)
    return render_template('index.html',pred='Emotion = {}'.format(result))

@app.route('/audio')
def play_audio():
    return send_file('G://BE_Project/uploaded_audio.wav', mimetype='audio/wav')

@app.route('/waveform')
def display_waveform():
    return send_file('G://BE_Project/waveform.png', mimetype='img/png')

@app.route('/mfcc')
def display_mfcc():
    return send_file('G://BE_Project/mfcc.png', mimetype='img/png')

@app.route('/pychart')
def display_pychart():
    return send_file('G://BE_Project/pychart.png', mimetype='img/png')

@app.route('/signin')
def signin():
    return render_template("signin.html")

if __name__=="__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.sleep(0.1))
    app.run(port=5000, debug=True)