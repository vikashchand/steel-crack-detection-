from flask import Flask, render_template, request, redirect, session, send_from_directory, url_for
import requests
import secrets

import tensorflow as tf
import numpy as np

from io import BytesIO
import os
from PIL import Image

from werkzeug.utils import secure_filename


app = Flask(__name__)


# setting up for uploading pic s
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

#define function for check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#lets add folder to app.config
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#dictionary for translating prediction results
dict_values = {0:'There are no cracks in the steel.',1:'There are cracks in the steel.'}


def load():
    global services 
    services = dict()
    
    services['model'] = tf.keras.models.load_model('resnet_trained.h5')
    

    
@app.route('/', methods =['GET','POST'])
@app.route('/home', methods =['GET','POST'])
def home_page():
    return render_template('index.html')


@app.route('/tool', methods =['GET','POST'])
def analyze():
    if request.method == 'POST':
        

        URL = request.form.get('url')
        file = request.files['file']

        if not URL:
            if file and allowed_file(file.filename):
                token_pic = secrets.token_hex(5)
                filename = str(token_pic) + str(secure_filename(file.filename))
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                img = Image.open(file)
                services['to_print'] = url_for('uploaded_file',filename=filename)
            else:
                print('Wrong format or not uploaded picture')
                return redirect('\home')
        else:
            response = requests.get(URL)
            img = Image.open(BytesIO(response.content))
            services['to_print'] = URL

         
        
        # preprocess it to the model format
        image_resized = img.resize((224,224))
        
        
        # predict image
        global prediction_
        prediction_ = services['model'].predict(np.expand_dims(np.array(image_resized, dtype='float32'), axis=0))

        # lets generate token for not overlapping users usage
            
        token = secrets.token_hex(5)
        return redirect(url_for('predict', token=token))

    else:
        return render_template('request.html')
            
@app.route('/<token>/prediction', methods = ['GET','POST'])
def predict(token):
    if request.method == 'POST':
        return redirect('/tool')
    else:
        result = dict_values[np.argmax(prediction_[0])]
        return render_template('prediction.html', prediction=result, img=services['to_print'])

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

def main():
    load()
    #this for local debugging
    #app.run(debug=True)
    
    #this for remote working
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

if __name__ == '__main__':
    main()
        
        
        
        
                
        
