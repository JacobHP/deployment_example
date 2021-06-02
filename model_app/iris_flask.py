import json
import waitress
from flask import Flask, request, Response, send_file
import pickle
import pandas as pd 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


#model_uri = r'file:///model/iris_model.pkl'
model_uri = r'model/iris_model.pkl'

with open(model_uri, 'rb') as f:
    model = pickle.load(f)

# flask app

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    X = pd.DataFrame(json.loads(request.data))
    y_hat = model.predict(X).tolist()
    response = json.dumps(y_hat)
    print(f'Prediction performed on {len(X)} samples')
    return response


def create_figure(X,y):
    fig, ax = plt.subplots()
    ax.bar(['setosa','versicolor','virginica'], y)
    ax.set_title(f'Prediction for {X.values}')
    ax.set_xlabel('Class')
    ax.set_ylabel('Probability')
    ax.set_ylim(0,1)
    ax.set_xticks([0,1,2])
    for i, v in enumerate(y):
        ax.text(i-0.1, v+.05, str(round(v,3)), fontweight='bold')
    plt.savefig('static/images/latest_plot.png')


@app.route('/predict_proba', methods=['POST'])
def predict_proba():
    X = pd.DataFrame(json.loads(request.data))
    y_hat = model.predict_proba(X)[0].tolist()
    create_figure(X, y_hat)
    return send_file('static/images/latest_plot.png', mimetype='img/gif')
    
waitress.serve(app, host='0.0.0.0', port=1337)




