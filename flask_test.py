import requests
import io
from PIL import Image
# add command line
dict_to_send = {'sepal_length': [0.2], 'sepal_width':[0.3], 'petal_length':[1], 'petal_width':[2]}

#res = requests.post('http://localhost:1337/predict_proba', json=dict_to_send)
res = requests.post('http://35.178.3.192:1337/predict_proba', json=dict_to_send)
#print('response from server:', res.content)
imageStream = io.BytesIO(res.content)
imageFile = Image.open(imageStream)
imageFile.save('result.png')
display(imageFile)