import json
import os
import pandas as pd

"""
1.column是'_'
2.column数量小于等于2
"""
def table():
    meta_table_path = "/data3/yss/tabular-data/meta_data.json"
    with open(meta_table_path)as f:
        meta_table = json.load(f)
    # with open("/data3/yss/tabular-data/deprecate_table.json")as f:
    #     deprecate_table = json.load(f)
        deprecate_table = {}
        n = 0
        for i,(table_id,values) in enumerate(meta_table.items()):
            if table_id in deprecate_table:
                continue
            for value in values:
                table_path = value["save_path_abs"]
                df = pd.read_csv(table_path)
                # print("table_path:",table_path)
                # if "_" in df.columns:
                #     print("table id:",table_id)
                #     print(df.head().to_markdown(index=False))
                #     print("*"*100)
                #     deprecate_table[table_id] = value
                    # exit()
                # 空字段
                if "Unnamed" in df.columns:
                    # print("table id:",table_id)
                    # print(df.head().to_markdown(index=False))
                    # print("*"*100)
                    deprecate_table[table_id] = value

                # 过滤列名大小写重复的  
                columns = [col.lower() for col in list(df.columns)]
                if len(list(set(columns))) != len(columns):
                    # print("table id:",table_id)
                    # print(df.head().to_markdown(index=False))
                    # print("*"*100)
                    deprecate_table[table_id] = value
                # 过滤小于2列
                if len(df.columns)<=2:
                    # print("table id:",table_id)
                    # print(df.head().to_markdown(index=False))
                    # print("*"*100)
                    deprecate_table[table_id] = values
                # 过滤小于1行
                if len(df)<=1:
                    # print("table id:",table_id)
                    # print(df.head().to_markdown(index=False))
                    # print("*"*100)
                    deprecate_table[table_id] = values
                # if len(df.columns)>=100:
                #     print("table id:",table_id)
                #     print(df.head(2).to_markdown(index=False))
                #     print("*"*100) 
                #     n+=1
                #     deprecate_table[table_id] = values
    print(len(deprecate_table))
    with open("deprecate_table.json","w")as f:
        json.dump(deprecate_table,f,ensure_ascii=False)
table()