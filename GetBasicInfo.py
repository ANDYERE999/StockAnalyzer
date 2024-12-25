import baostock as bs
import pandas as pd
import datetime
import json
import csv
# 打开JSON文件
with open('./files/pyData/GetBasicInfo.json',encoding='utf-8') as file1:
    # 加载JSON数据
    data = json.load(file1)

# 打印JSON数据
print(data)
processed_data=list(data["stock_name"])

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

    result = pd.DataFrame(data_list, columns=rs.fields)
bs.logout()
scvFileName = "股票基本信息" + str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')) + ".csv"
path_of_csv1 = "./files/QTRequireData/GetBasicInfo.csv"
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