import json

with open('./data/20ng/20ng', 'r',encoding='utf-8') as inp_txt:
    all_lines = inp_txt.readlines()[:-1]
    all_txt=[]
    all_lable=[]
    for id in all_lines:
        data_dic=json.loads(id)
        all_txt.append(data_dic["text"])
        all_lable.append(data_dic["cluster"])
    print(min(all_lable))