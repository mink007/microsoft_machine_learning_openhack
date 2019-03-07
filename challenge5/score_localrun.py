#!/usr/bin/python

import os
from keras.models import Sequential, load_model
from PIL import Image
import numpy as np
import sys
import requests

imURL = sys.argv[1]
#imURL = "http://content.backcountry.com/images/items/900/MNT/MNT0012/MORBLUOR.jpg"

response = requests.get(imURL, stream=True)
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
print(prediction)

label = categories[np.argmax(prediction)]
print("label: " + label) 
