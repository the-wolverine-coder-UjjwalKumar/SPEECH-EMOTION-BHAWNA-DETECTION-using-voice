from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from utils import extract_feature,convert
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

app = Flask(__name__)

## Configure upload location for audio
app.config['UPLOAD_FOLDER'] = "./audio"

## Route for home page
@app.route('/')
def home():
    return render_template('main.html',value="")


## Route for results
@app.route('/results', methods = ['GET', 'POST'])
def results():
    """
    This route is used to save the file, convert the audio to 16000hz monochannel,
    and predict the emotion using the saved binary model
    """
    if not os.path.isdir("./audio"):
        os.mkdir("audio")
    if request.method == 'POST':
        try:
          f = request.files['file']
          filename = secure_filename(f.filename)
          f.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        except:
          return render_template('main.html', value="")

    wav_file_pre  = os.listdir("./audio")[0]
    wav_file_pre = f"{os.getcwd()}/audio/{wav_file_pre}"
    wav_file = convert(wav_file_pre)
    os.remove(wav_file_pre)
    model = pickle.load(open(f"{os.getcwd()}/model.model", "rb"))
    x_test =extract_feature(wav_file)
    y_pred=model.predict(np.array([x_test]))
    os.remove(wav_file)
    print(y_pred)
    return render_template('main.html', value=y_pred[0])
    
@app.route("/aboutus")
def aboutus():
	return render_template("aboutus.html")

@app.route("/contact")
def contact():
	return render_template("contact.html")

@app.route("/blog")
def blog():
	return render_template("blog.html")

@app.route("/test")
def testVoice():
    return render_template("test.html")

if __name__ == '__main__':
	app.run(debug=1)