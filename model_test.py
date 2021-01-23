# load_modelをインポートする
import pickle
import pandas as pd

#modelへ保存データを読み込み
# model = pd.read_pickle("model.pickle") 
model = {}
with open("model.pickle", mode="rb") as f:
    model = pickle.load(f)

#評価データ(ここは自分で好きな値を入力)
weather = [[9,0,7.9,6.5]]    #気温、降水量、日照時間、湿度、天気

#predict関数で、評価データの天気を予測
ans = model.predict(weather)

if ans == 0:
    print("晴れです")
if ans == 1:
    print("曇りです")
if ans == 2:
    print("雨です")
