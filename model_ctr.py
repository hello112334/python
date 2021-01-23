import numpy as np
from sklearn import svm
import pickle

def main():
    print(" ----- main ")

    #csvファイルの読み込み
    #気温、降水量、日照時間、湿度、天気
    npArray = np.loadtxt("./data/data.csv", delimiter = ",", dtype = "float")

    # 説明変数の格納
    x = npArray[:, 0:4]

    #目的変数の格納
    y = npArray[:, 4:5].ravel()

    #学習手法にSVMを選択
    model = svm.SVC()

    #学習
    model.fit(x,y)

    # #学習モデルの保存
    with open('model.pickle', mode='wb') as f:
        pickle.dump(model,f,protocol=2)

    # #評価データ(ここは自分で好きな値を入力)
    # weather = [[9,0,7.9,6.5]] 

    # #predict関数で、評価データの天気を予測
    # ans = model.predict(weather)

    # if ans == 0:
    #     print("晴れです")
    # if ans == 1:
    #     print("曇りです")
    # if ans == 2:
    #     print("雨です")    

# 本体
if __name__ == '__main__':   
    print(" ------------ START ------------ ")

    # メイン処理
    main() 
    
    print(" ------------ END ------------ ")