import numpy as np
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import pickle
import math

def main():
    print(" ----- main ")

    #csvファイルの読み込み
    #気温、降水量、日照時間、湿度、天気
    npArray = np.loadtxt("./data/data.csv", delimiter = ",", dtype = "float")

    # 説明変数の格納
    x = npArray[:, 0:4]

    #目的変数の格納
    y = npArray[:, 4:5].ravel()

    # 学習手法を選択
    # model = svm.SVC()
    # model = svm.SVR()
    # model = svm.LinearSVC(max_iter=10000)
    # model = svm.LinearSVR(max_iter=10000)
    # model = LinearRegression()
    model= LogisticRegression(max_iter=10000)

    #学習
    model.fit(x,y)

    # 学習モデルの保存
    with open('model.pickle', mode='wb') as f:
        pickle.dump(model,f,protocol=2)

    # 評価データ(ここは自分で好きな値を入力)  
    # weather = [[16,0,12.9,32.5]] # 晴れ
    # weather = [[11,0,5.0,34.0]] # 曇り
    weather = [[6.6,1.5,0.5,68.5]] # 雨

    # predict関数で、評価データの天気を予測
    # ans = round(float(model.predict(weather)))
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

# 本体
if __name__ == '__main__':   
    print(" ------------ START ------------ ")

    # メイン処理
    main() 
    
    print(" ------------ END ------------ ")