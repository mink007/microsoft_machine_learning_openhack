from flask import Flask, jsonify, request


#import os
from keras.models import Sequential, load_model
from PIL import Image
import numpy as np
import sys
import requests

app = Flask(__name__)


@app.route('/api')
def response():
    # here we want to get the value of user (i.e. ?url=some-value)
    # "for tesing example http://127.0.0.1:5000/api?url=http://content.backcountry.com/images/items/900/MNT/MNT0012/MORBLUOR.jpg"
    url = request.args.get('url')
    response = requests.get(url, stream=True)
    response.raw.decode_content=True

    img=Image.open(response.raw)
    img.thumbnail((128,128))

    imgNew = Image.new("RGB", (128,128), "white")
    offsetX = int((128 - img.width)/2)
    offsetY = int((128 - img.height)/2)
    imgNew.paste(img, (offsetX, offsetY))
    testImg = np.reshape(imgNew,[1,128,128,3])
    #print(testImg)
    model = load_model('objectmodelchal5.h5')
    categories = {0: 'axes', 1: 'boots', 2: 'carabiners', 3: 'crampons', 4: 'gloves', 5: 'hardshell_jackets', 6: 'harnesses',
              7: 'helmets', 8: 'insulated_jackets', 9: 'pulleys', 10: 'rope', 11: 'tents'}
    prediction = model.predict(testImg)
    label = categories[np.argmax(prediction)]
    #print("label" + label) 
    

    return "Your label is : " + label


if __name__ == '__main__':
    debug = True  # set to true for hot-reload
    app.run(debug=debug)
