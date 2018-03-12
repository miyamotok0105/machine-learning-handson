from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
 
'''
データ準備
'''
np.random.seed(0) # 乱数を固定値で初期化し再現性を持たせる
 
iris = datasets.load_iris()
X = iris.data
T = iris.target
 
T = np_utils.to_categorical(T) # 数値を、位置に変換 [0,1,2] ==> [ [1,0,0],[0,1,0],[0,0,1] ]
train_x, test_x, train_t, test_t = train_test_split(X, T, train_size=0.8, test_size=0.2) # 訓練とテストで分割
 
'''
モデル作成
'''
model = Sequential()
model.add(Dense(input_dim=4, units=3))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1))
 
'''
トレーニング
'''
# 損失関数がより小さくなったときだけモデルをファイル保存するコールバック
checkpointer = ModelCheckpoint(filepath = "iris.h5", save_best_only=True)
 
# 学習成果をモニターしながら fit させるためにテストデータとその正解を validation_data として付加
model.fit(train_x, train_t, epochs=40, batch_size=10, validation_data=(test_x, test_t), callbacks=[checkpointer])
 
'''
新たに作ったモデルにファイルから読み込む
'''
app = load_model("iris.h5")
 
'''
ファイルから作ったモデルで分類する
'''
Y = app.predict_classes(test_x, batch_size=10)
 
'''
結果検証
'''
_, T_index = np.where(test_t > 0) # to_categorical の逆変換
print()
print('RESULT')
print(Y == T_index)

