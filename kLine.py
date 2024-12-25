import baostock as bs
import pandas as pd
import json
import datetime
import csv


with open('./files/pyData/kLine.json',encoding='utf-8') as file1:
    # 加载JSON数据
    data = json.load(file1)
#print(data)
aimedStockList=data["stock_id"]
aimedStockFields=data["fields"]
aimedStockStartDate=data["start_date"]
aimedStockEndDate=data["end_date"]
if aimedStockEndDate=="":
    aimedStockEndDate=str(datetime.datetime.now().strftime('%Y-%m-%d'))
if aimedStockStartDate=="":
    aimedStockStartDate="2024-01-01"
###print("------------------------------")
#print(data)
###print(aimedStockList)
###print(aimedStockFields)
###print(aimedStockStartDate)
###print(aimedStockEndDate)###
#### 登陆系统 ####
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

#### 获取沪深A股历史K线数据 ####
# 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
# 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
# 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg


#### 打印结果集 ####
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

#### 登出系统 ####
bs.logout()

#### 结果集输出到csv文件 ####
scvFileName = "股票K线信息" + str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')) + ".csv"
path_of_csv1 = "./files/QTRequireData/kLine.csv"
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

