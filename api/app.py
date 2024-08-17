#
import json
import joblib
import numpy as np
from flask import Flask, jsonify, request, render_template

#


#
app = Flask(__name__)


@app.route("/")
def hello():
    return "Hoilaaaaaaaaa!"


@app.route('/view/m1', methods=['GET'])
def m1_view():
    return """
<body style="margin:0px;padding:0px;overflow:hidden">
    <iframe src="http://localhost:8080/" frameborder="0" style="overflow:hidden;height:100%;width:100%" height="100%" width="100%"></iframe>
</body>
    """


@app.route('/predict/m1', methods=['POST'])
def m1_call():
    try:
        test_json = request.get_json()
        data = {'BTC-USD': [],
                'DOGE-USD': [],
                'ETH-USD': [],
                'LTC-USD': [],
                'XRP-USD': []}

        for dic in test_json:
            target = dic['target']
            with open('model/{0}/m1_features.json'.format(target)) as f:
                features = json.load(f)
                row = []
                for feature in features['features']:
                    row.append(dic[feature])
                data[target].append(row)
        # load model
        pred_dict = {}
        for key in data.keys():
            pdata = np.array(data[key])
            if pdata.shape[0] > 0:
                model = joblib.load('model/{0}/m1.pkl'.format(key))
                y_pred = model.predict(pdata)
                for i, pred in enumerate(y_pred):
                    pred_dict['prediction_{0}_{1}'.format(key, i)] = float(pred)
        responses = jsonify(predictions=pred_dict)
        responses.status_code = 200
    except Exception as e:
        print(e)
        responses = jsonify(predictions={'error': str(e)})
        responses.status_code = 404
    return responses


"""
waitress-serve --listen=localhost:8000 app:app
"""

"""
import json
import pandas
import urllib3

g = 'C:/Users/Edward/Desktop/ex_req.xlsx'
rg = pandas.read_excel(g)

rsg = []
rgg = {'target': 'BTC-USD'}
for j in range(rg.shape[0]):
    rgg[rg.values[j, 0]] = rg.values[j, 1]
rsg.append(rgg)

encoded_body = json.dumps(rsg)

http = urllib3.PoolManager()
"""
"""
r = http.request('POST', 'http://localhost:8000/predict',
                 headers={'Content-Type': 'application/json'},
                 body=encoded_body)
"""
"""
r = http.request('POST', 'http://localhost:8080/predict',
                 headers={'Content-Type': 'application/json'},
                 body=encoded_body)
"""
