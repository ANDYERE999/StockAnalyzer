import baostock as bs
import pandas as pd
import datetime
import json
import csv
import numpy as np
# 打开JSON文件
with open('./files/pyData/getKline+.json',encoding='utf-8') as file1:
    # 加载JSON数据
    data = json.load(file1)

# 打印JSON数据
print(data)
processed_data=list(data["stock_name"])
stock_nameCount=len(processed_data)

lg=bs.login()
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)
data_list = []
for i in range(len(processed_data)):
    stockName=processed_data[i]
    rs=bs.query_stock_basic(code_name=stockName)
    #print('query_stock_basic respond error_code:'+rs.error_code)
    #print('query_stock_basic respond  error_msg:'+rs.error_msg)
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result1 = pd.DataFrame(data_list, columns=rs.fields)
bs.logout()

path_of_csv1 = "./files/pyData/getKline+_mid.csv"
result1.to_csv(path_or_buf=path_of_csv1,encoding="gbk",index=False)
print('数据已经保存到'+path_of_csv1)

###从csv中读取所需要的股票代码
data=pd.read_csv(path_of_csv1,encoding='gbk',usecols=['code'])
data=np.array(data)
stockCodeList=[]
for i in range(len(data)):
    stockCodeList.append(data[i][0])
print(stockCodeList)

###获取K线
# 打开JSON文件
with open('./files/pyData/getKline+.json',encoding='utf-8') as file1:
    # 加载JSON数据
    data = json.load(file1)
aimedStockList=stockCodeList
aimedStockFields=data["fields"]
aimedStockStartDate=data["start_date"]
aimedStockEndDate=data["end_date"]
lg = bs.login()
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)
data_list = []

for code in aimedStockList:
    rs = bs.query_history_k_data_plus(str(code),
                                      fields=aimedStockFields,
                                      start_date=aimedStockStartDate, end_date=aimedStockEndDate,
                                      frequency="d", adjustflag="3")
    #print('query_history_k_data_plus respond error_code:' + rs.error_code)
    #print('query_history_k_data_plus respond  error_msg:' + rs.error_msg)
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
result = pd.DataFrame(data_list, columns=rs.fields)
rowCount=result.shape[0]
#print(rowCount)
rowCount=int(rowCount/stock_nameCount)
stockNameRepList=[]
for j in range(stock_nameCount):
    for i in range(rowCount):
        stockNameRepList.append(processed_data[j])
#print(stockNameRepList)
result.insert(0,'codeName',stockNameRepList)
#### 登出系统 ####
bs.logout()

#### 结果集输出到csv文件 ####
scvFileName = "股票K线信息" + str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')) + ".csv"
path_of_csv1 = "./files/QTRequireData/kLine+.csv"
result.to_csv(path_or_buf=path_of_csv1,encoding="gbk",index=False)
path_of_csv2="./UserDataExcel/"+scvFileName
result.to_csv(path_or_buf=path_of_csv2,encoding="gbk",index=False)

#将csv转换为txt以供QT读取
csv_file_path=path_of_csv1
txt_file_path=csv_file_path.replace('.csv','.txt')
with open(csv_file_path, 'r', encoding='gbk') as file:
    csv_reader=csv.reader(file)
    for row in csv_reader:
        with open(txt_file_path, 'a', encoding='utf-8') as txt_file:
            txt_file.write(','.join(row)+'\n')
print('数据已经保存到'+path_of_csv1)

