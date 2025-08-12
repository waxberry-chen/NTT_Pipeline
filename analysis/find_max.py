import os
import pandas as pd
import random
# # 指定包含所有txt文件的目录
# directory = 'D:\\pqc\\side_channel_analysis\\PWM_Pipeline_Data'
# file_extension = '.txt'  # 文件扩展名

# # 构造文件名列表，按数字从小到大的顺序
# file_names = [f'{i}{file_extension}' for i in range(1,3327)]

# # 用于保存所有文件的最大值
# all_max_values = []
line_number=600
# for filename in file_names:
#     file_path = os.path.join(directory, filename)

#     # 读取txt文件并将数据按行划分
#     with open(file_path, 'r') as file:
#         lines = file.readlines()

#     # 将数据分组为line_number行一组，每组找到最大值并保存到列表
#     max_values = []
#     for i in range(0, len(lines), line_number):
#         group = lines[i:i+line_number]
#         values = [int(float(val)) for val in group]  # 将字符串转换为整数
#         max_values.append(max(values))

#     # 将每个文件的最大值列表保存到总的最大值列表中
#     all_max_values.append(max_values)
# # 转置以确保每个文件的最大值作为一列
# all_max_values_transposed = list(map(list, zip(*all_max_values)))
# # 创建DataFrame
# df = pd.DataFrame(all_max_values_transposed, columns=[f'key{i}' for i in range(len(all_max_values))])

# # 将所有最大值保存到Excel文件
# output_file_path = 'output.xlsx'
# df.to_excel(output_file_path, index=False)

# print(f'Data saved to {output_file_path}')

# # 读取Excel文件
# excel_file_path = 'output.xlsx'
# df = pd.read_excel(excel_file_path, header=None, skiprows=1)

# # 将数据保存为文本文件，以空格分隔不同列
# text_file_path = 'output.txt'
# df.to_csv(text_file_path, sep=' ', index=False, header=False)
# print(f'Data saved to {text_file_path}')
def calculate_r(H, T):
    h_bar = sum(H) / len(H)
    t_bar = sum(T) / len(T)

    h_sigma = sum([(h - h_bar) ** 2 for h in H])
    t_sigma = sum([(t - t_bar) ** 2 for t in T])
    
    ht_sum = sum([(H[i] - h_bar) * (T[i] - t_bar) for i in range(len(H))])

    r = ht_sum / (h_sigma * t_sigma) ** 0.5
    return r

# 读取txt文件并将每行数据保存到列表中，同时记录原始行号
original_lines = []
with open('output.txt', 'r') as file:
    for i, line in enumerate(file):
        original_lines.append((i, line.strip()))

# 打乱数据顺序
random.shuffle(original_lines)
number=1
empty_df = pd.DataFrame()
seen_numbers = set()
while number <= 1500:
    if round(number) not in seen_numbers:
        # 选择打乱后的前number行数据
        shuffled_lines = original_lines[:round(number)]

        # 将打乱后的前number行数据写回到txt文件，并记录对应关系
        with open('shuffled_file.txt', 'w') as file:
            for i, line in shuffled_lines:
                file.write(f"{i+1} {line}\n")
        import numpy as np

        # 读取txt文件并解析为数组
        def read_txt_file(filename):
            data = []
            with open(filename, 'r') as file:
                for line in file:
                    # 分割每行数据，以空格为分隔符，并转换为整数
                    values = [int(value) for value in line.strip().split()]
                    # 将每行数据添加到data
                    data.append(values)
            return np.array(data)  # 转换为NumPy数组

        # 文件路径
        file_path = 'shuffled_file.txt'

        # 读取txt文件并解析为数组
        data_array = read_txt_file(file_path)

        # 输出数组的形状
        print('Shape of the data array:', data_array.shape)



        # 读取文本文件并分割数据
        with open('y.txt', 'r') as file:
            lines = file.readlines()

        # 将数据转换为二维数组，每line_number行为一个子数组
        grouped_data = [lines[i:i + line_number] for i in range(0, len(lines), line_number)]

        # 找出每组的最大值
        max_values = [max(map(int, group)) for group in grouped_data]

        # 将最大值保存到数组
        max_values_array = np.array(max_values)
        #target这是猜测key对应一千个点的用来算互相关的
        target_cor = max_values_array[data_array[:,0]-1]

        correlations = []
        for i in range(data_array.shape[1]-1):
            r = calculate_r(target_cor, data_array[:, i+1])
            correlations.append(r)
        # Convert correlations to a pandas Series
        # 找到最大值及其索引
        max_value = np.max(correlations)
        max_index = np.argmax(correlations)
        row_number=f'row_{round(number)}'    
        # 输出最大值及其索引
        print("max:", max_value)
        print("index:", max_index)
        df = pd.DataFrame({row_number: correlations})
        # 将现有的 DataFrame 和新的 DataFrame 合并
        empty_df = pd.concat([empty_df, df], axis=1)
    seen_numbers.add(round(number))
    number *= 1.1
output_file_path = 'find_y_1000.xlsx'
empty_df.to_excel(output_file_path, index=False)   
#根据x—txt文件和模板，找到对应的索引