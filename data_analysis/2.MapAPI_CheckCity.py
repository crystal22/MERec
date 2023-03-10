import pandas as pd
import requests
import json
import numpy as np

data = pd.read_csv('./Calgary.csv', encoding='ISO-8859-1')
print(data.shape)
data = data.drop_duplicates()
print(data.shape)

for index, row in data.iterrows():
    longitude = row[1]
    dimension = row[2]
    #通过经纬坐标查询城市
    key = 'GjG3XAdmywz7CyETWqHwIuEC6ZExY6QT'
    r = requests.get(url='http://api.map.baidu.com/geocoder/v2/', params={
    'location':'%.6f m,%.6f m'%(longitude,dimension),'ak':key,'output':'json'})
    result = r.json()
    #！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    city = result['result']['addressComponent']['city']
    #print(city)
    if city == "Calgary":
        print("---")
    else:
        #print(index, row[0], row[1], row[2], row[3], city)
        data = data.drop([index])
       # print(data)
print(data.shape)
data.to_csv('Calgary_new.csv',index=False,sep=',')


