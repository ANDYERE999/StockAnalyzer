import baostock as bs
import pandas as pd
import json
import datetime
import time
import random
import os
print("#" * 60)
print("#" + " " * 20 + "即将更新数据库" + " " * 20 + "#")
print("#" * 60)
print("")

with open('./files/config.json', encoding='utf-8') as file1:
    data = json.load(file1)

aimedStockList = data["stock_id"]
aimedStockFields = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"
aimedStockEndDate = datetime.datetime.now().strftime('%Y-%m-%d')
configDataDate=str(data["databaseTime"])

print("#" * 60)
print("#"+" " * 20 + "本地数据库截止日期：    " + configDataDate + " " * 20 + "#")
print("#" * 60)
print("")

lg = bs.login()
print("\033[F\033[F", end='')  # 移动光标到前两行
print("\033[K", end='')  # 清除当前行
print("\033[K", end='')  # 清除当前行
print("#" * 60)
#print('login respond error_code:' + lg.error_code)
#print('login respond  error_msg:' + lg.error_msg)

dataGet = 0

for code in aimedStockList:
    try:
        data_list = []
        dataGet += 1

        print(f"#正在获取第 {dataGet}/1692 个股票数据：{code}#",end="\r")

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
            print(f"#{code} 数据已是最新，无需更新。#",end="\r")
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
            ###print(f"{code} 数据已更新。")
        else:
            pass
            ###print(f"{code} 无新增数据。")
        sleepTime = random.uniform(0.1, 0.3)
        #print("Sleeping for " + str(sleepTime) + " seconds...")
        #time.sleep(sleepTime)
    except Exception as e:
        print(f"#Warning!{code} Error：{e}     #")
        if (0==0):
            print(f"#推测原因1：该股票已经退市#")
            print(f"#推测原因2：网络波动/服务器未响应#")

bs.logout()

data["databaseTime"] = aimedStockEndDate
with open('./files/config.json', 'w', encoding='utf-8') as file1:
    json.dump(data, file1, ensure_ascii=False, indent=4)

print("#" * 60)
print("#" + " " * 20 + "数据库更新完成" + " " * 20 + "#")
print("#" * 60)
