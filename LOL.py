import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv(r"E:\datesets\league-of-legends-diamond-ranked-games-10-min\high_diamond_ranked_10min.csv")

data.drop(["gameId"
           ,"blueWardsPlaced"
           ,"blueWardsDestroyed"
           #,"blueAssists"
           #,"blueEliteMonsters"
           ,"blueAvgLevel"
           ,"blueTotalJungleMinionsKilled"
           ,"blueCSPerMin"
           ,"blueGoldPerMin"
           ,"blueGoldDiff"
           ,"blueExperienceDiff"
           ,"blueTotalExperience"
           ,"redWardsPlaced"
           ,"redWardsDestroyed"
           ,"redFirstBlood"
           ,"redKills"
           ,"redDeaths"
           #,"redAssists"
           #,"redEliteMonsters"
           ,"redAvgLevel"
           ,"redTotalJungleMinionsKilled"
           ,"redGoldDiff"
           ,"redExperienceDiff"
           ,"redTotalExperience"
           ,"redCSPerMin"
           ,"redGoldPerMin"],inplace = True ,axis = 1)

x = data.iloc[:,data.columns != "blueWins"] #数据的特征
y = data.iloc[:,data.columns == "blueWins"] #数据的标签

Xtrain, Xtest, Ytrain, Ytest = train_test_split(x,y,test_size = 0.3)

for i in [Xtrain, Xtest, Ytrain, Ytest]:
    i.index = range(i.shape[0])

lol = DecisionTreeClassifier(random_state=0
                            ,criterion = 'entropy'
                            ,max_depth = 5
                            ,min_impurity_decrease = 0.0
                            ,min_samples_leaf = 46
                            ,splitter ='best')#实例化
lol = lol.fit(Xtrain,Ytrain)

import csv
lis = ['blueFirstBlood'
                 ,'blueKills'
                 ,'blueDeaths'
                 ,'blueAssists'
                 ,'blueEliteMonsters'
                 ,'blueDragons'
                 ,'blueHeralds'
                 ,'blueTowersDestroyed'
                 ,'blueTotalGold'
                 ,'blueTotalMinionsKilled'
                 ,'redAssists'
                 ,'redEliteMonsters'
                 ,'redDragons'
                 ,'redHeralds'
                 ,'redTowersDestroyed'
                 ,'redTotalGold'
                 ,'redTotalMinionsKilled']
header = np.array(lis)

lol_data = []
for i in range(17):
    print("Please Iuput "+lis[i]+" : ",end='')
    s = int(input())
    lol_data.append([s])
    
lol_data = np.array(lol_data)

with open('lol.csv','w',newline='') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(header)
    f_csv.writerows(lol_data.T)

want_data = pd.read_csv(r'lol.csv')

print()
if lol.predict(want_data) == 1:
    print("BlueTeamWin, Probability: "+str(lol.predict_proba(want_data)[0][1]))
else:
    print("BlueTeamLose, Probability: "+str(lol.predict_proba(want_data)[0][0]))

import os
os.remove('lol.csv')

