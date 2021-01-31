# load_modelをインポートする
import pickle
import pandas as pd

#modelへ保存データを読み込み
# model = pd.read_pickle("model.pickle") 
model = {}
with open("model.pickle", mode="rb") as f:
    model = pickle.load(f)

# 評価データ(ここは自分で好きな値を入力)
# weather = [[16,0,12.9,32.5]] # 晴れ
weather = [[11,0, 5.0,34.0]] # 曇り
# weather = [[6.6,1.5,0.5,68.5]] # 雨

# predict関数で、評価データの天気を予測
# ans = model.predict(weather)
ans = float(model.predict(weather))
print("Result: {0}".format(float(model.predict(weather))))

if ans >= 0 and ans < 0.5:
    print("晴れです")
elif ans >= 0.5 and ans < 1.5:
    print("曇りです")
elif ans >= 1.5:
    print("雨です")      
# else:
#     print("None")
