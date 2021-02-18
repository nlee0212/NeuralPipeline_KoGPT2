import json

with open('train2_v4.json',"r",encoding="utf-8") as f1:
    data1 = json.load(f1)

with open('test2_v4.json',"r",encoding="utf-8") as f2:
    data2 = json.load(f2)

with open('total2_v4.json', "w", encoding = "utf-8") as new_file:
    json.dump({**data1,**data2}, new_file,indent=4)
