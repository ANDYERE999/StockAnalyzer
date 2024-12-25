import baostock as bs
import pandas as pd
import json
import datetime
import time
import random
import os

with open('./files/config.json', encoding='utf-8') as file1:
    data = json.load(file1)

aimedStockList = ["sh.600000", "sh.600004", "sh.600006"]
aimedStockFields = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"
aimedStockEndDate = datetime.datetime.now().strftime('%Y-%m-%d')

lg = bs.login()
print('login respond error_code:' + lg.error_code)
print('login respond  error_msg:' + lg.error_msg)

dataGet = 0

for code in aimedStockList:
    data_list = []
    dataGet += 1

    # 读取对应的 CSV 文件，删除空行
    csv_file_path = f'./files/db/{code}.csv'
    if os.path.exists(csv_file_path):
        df_existing = pd.read_csv(csv_file_path, encoding='gbk').dropna(how='all')  # 删除空行
        # 获取文件中最新的日期
        if df_existing.empty:
            last_date = data["databaseTime"]
        else:
            last_date = df_existing['date'].max()
    else:
        df_existing = pd.DataFrame()
        last_date = data["databaseTime"]

    # 计算需要更新的开始日期
    aimed_date = datetime.datetime.strptime(last_date, "%Y-%m-%d")
    aimed_date += datetime.timedelta(days=1)
    aimedStockStartDate = aimed_date.strftime("%Y-%m-%d")

    # 如果开始日期大于结束日期，说明不需要更新
    if aimedStockStartDate > aimedStockEndDate:
        print(f"{code} 数据已是最新，无需更新。")
        continue

    # 获取新数据
    rs = bs.query_history_k_data_plus(
        str(code),
        fields=aimedStockFields,
        start_date=aimedStockStartDate,
        end_date=aimedStockEndDate,
        frequency="d",
        adjustflag="3"
    )

    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())

    result = pd.DataFrame(data_list, columns=aimedStockFields.split(","))

    # 删除 result 中的表头行（如果存在）
    result = result[result['date'] != 'date']

    if not result.empty:
        # 将新数据追加到 CSV 文件末尾
        result.to_csv(csv_file_path, mode='a', header=False, index=False, encoding='gbk')
        print(f"{code} 数据已更新。")
    else:
        print(f"{code} 无新增数据。")

    sleepTime = random.uniform(0.1, 0.3)
    print("Sleeping for " + str(sleepTime) + " seconds...")
    time.sleep(sleepTime)

bs.logout()

data["databaseTime"] = aimedStockEndDate
with open('./files/config.json', 'w', encoding='utf-8') as file1:
    json.dump(data, file1, ensure_ascii=False, indent=4)