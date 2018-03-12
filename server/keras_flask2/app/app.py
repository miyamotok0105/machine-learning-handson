from flask import Flask, request
import numpy as np
import keras.models
import tensorflow as tf
 
app = Flask(__name__)
 
# model and backend graph must be created on global
global model, graph
model = keras.models.load_model("../model/iris.h5")
graph = tf.get_default_graph()
 
@app.route('/', methods=['GET'])
def root():
    names = [
    'Iris-Setosa セトナ ヒオウギアヤメ',
    'Iris-Versicolour バーシクル ブルーフラッグ',
    'Iris-Virginica バージニカ'
    ]
     
    sl = sw = pl = pw = 0.0
    sl = request.args.get('sepal_l')
    sw = request.args.get('sepal_w')
    pl = request.args.get('petal_l')
    pw = request.args.get('petal_w')
    parm = np.array([[sl, sw, pl, pw]])
     
    with graph.as_default(): # use the global graph
        predicted = model.predict_classes(parm)
        return names[predicted[0]]
 
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

