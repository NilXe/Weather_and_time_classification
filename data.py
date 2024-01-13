import json  

data_path = r'C:\Users\Administrator\Desktop\Deep\Weather and time classification\data\train_dataset'
# # 打开文件并读取内容  
# with open(data_path+'\\train.json', 'r') as f:  
#     data = json.load(f)  

# period_dic = {'Morning': '1', 'Afternoon': '2', 'Dawn': '3', 'Dusk': '4'}
# weather_dic = {'Cloudy': '1', 'Sunny': '2', 'Rainy': '3'}

# dic = {}
# num = 1
# for i in period_dic.keys():
#     for n in weather_dic.keys():
#         print(i, "+", n)
#         key = i + "+" + n
#         dic[key] = num
#         num+=1
# print(dic)

# json_data = json.dumps(dic)  

# # 将JSON字符串写入文件  
# with open(r'C:\Users\Administrator\Desktop\Deep\Weather and time classification\data\train_dataset\label.json', 'w') as f:  
#     f.write(json_data)

# print('转换完成！')



# # 打印数据  
# lst = data['annotations']
# data_lst = []
# for i in lst:
#     path = data_path + i['filename']
#     period = i['period']
#     weather = i['weather']
#     label = int(period_dic[period] + weather_dic[weather])
#     dic = {'path':path, 'label':label}
#     data_lst.append(dic)


# new_data = {'annotations':data_lst}
# import json  


# # 将字典转换为JSON字符串  
# json_data = json.dumps(new_data)  

# # 将JSON字符串写入文件  
# with open(r'C:\Users\Administrator\Desktop\Deep\Weather and time classification\data\train_dataset\data.json', 'w') as f:  
#     f.write(json_data)

# print('转换完成！')

# label_dic = {
#     "Morning+Cloudy": 1,
#     "Morning+Sunny": 2,
#     "Morning+Rainy": 3,
#     "Afternoon+Cloudy": 4,
#     "Afternoon+Sunny": 5,
#     "Afternoon+Rainy": 6,
#     "Dawn+Cloudy": 7,
#     "Dawn+Sunny": 8,
#     "Dawn+Rainy": 9,
#     "Dusk+Cloudy": 10,
#     "Dusk+Sunny": 11,
#     "Dusk+Rainy": 12
# }
p_label_dic = {
    "Morning": 0,
    "Afternoon": 1,
    "Dawn": 2,
    "Dusk": 3
}
w_label_dic = {
    "Cloudy": 0,
    "Sunny": 1,
    "Rainy": 2
}
with open(data_path+'\\train.json', 'r') as f:  
    data = json.load(f)  

lst = data['annotations']

data_lst = []
for i in lst:
    path = i['filename']
    period = i['period']
    weather = i['weather']

    dic = {'path':path, 'p_label':p_label_dic[period], 'w_label':w_label_dic[weather]}
    data_lst.append(dic)


new_data = {'annotations': data_lst}
import json  


# 将字典转换为JSON字符串  
json_data = json.dumps(new_data)  

# 将JSON字符串写入文件  
with open(r'C:\Users\Administrator\Desktop\Deep\Weather and time classification\data\train_dataset\data.json', 'w') as f:  
    f.write(json_data)

print('转换完成！')

# data_json = json.load(open(r'C:\Users\Administrator\Desktop\Deep\Weather and time classification\data\train_dataset\data.json'))
# data_lst = data_json['annotations']

# for i in data_lst:
#     i['label'] = i['label'] - 1

# new_data = {'annotations':data_lst}
# import json  
# # 将字典转换为JSON字符串  
# json_data = json.dumps(new_data)  
# # 将JSON字符串写入文件  
# with open(r'C:\Users\Administrator\Desktop\Deep\Weather and time classification\data\train_dataset\data01.json', 'w') as f:  
#     f.write(json_data)

# print('over')