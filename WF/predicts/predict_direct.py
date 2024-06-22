import numpy as np  

# 读取 ground_name.txt 文件中的文件名  
with open("ground_name.txt", "r") as file:  
    file_names = [line.strip() for line in file]  
  
# 读取 predicts_probability.txt 文件中的预测概率  
with open("predicts_probability.txt", "r") as file:  
    probabilities = [float(line.strip()) for line in file]  

assert len(probabilities) % 3 == 0, "The number of probabilities must be divisible by 3."  
 
# 将 probabilities 列表重塑为二维数组，每行三个值  
similarity = np.array(probabilities).reshape(-1, 3)  
# print(similarity)

value = []  
index = []  
  
# 计算每组数据的最大概率值和其索引  
for group in similarity:  
    group_max = np.max(group)  
    value.append(group_max)  
    index.append(np.argmax(group))  
# print(index)
# 分离出索引不等于 2 的文件和相似度矩阵  
abn = [i for i, idx in enumerate(index) if idx != 1]  
asim = similarity[abn]  
abn_file_names = [file_names[i] for i in abn]  

print(len(abn_file_names))

np.savetxt('asim.txt', asim, delimiter='\t')

with open('abn_file_names.txt', 'w') as f:
    for name in abn_file_names:
        f.write(name + '\n')
  
print("Data saved successfully.")