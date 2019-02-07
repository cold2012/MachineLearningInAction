import numpy as np
from math import log
import operator
import random

POS_SAM = 'Yes'
NEG_SAM = 'No'
TRAIN_NUM = 10

def read_data(file_name):
    data_set   = []
    label_list = []
    data_file  = open(file_name)
    while True:
        data_line = data_file.readline()
        if 0 == len(data_line):
            # 读不到文本
            break
        
        data_line = data_line.strip()
        if 0 == len(data_line) or '#' == data_line[0]:
            # 注释或者空行
            continue
        
        # split默认删除所有空格
        data_list = data_line.split()
        if data_list[0].isalpha():
            # 取特征名，去掉Id
            label_list = data_list[1:]
            continue
        
        # 去掉Id
        data_set.append(data_list[1:])
    
    data_file.close()
    # train_num = int(len(data_set) * 0.8)
    train_num = TRAIN_NUM
    train_set = []
    test_set  = []
    # 随机8/2分取训练集和测试集
    '''
    for i in range(train_num):
        j = random.randint(0, len(data_set) - 1)
        train_set.append(data_set[j])
        del(data_set[j])
        i += 1
    '''

    train_set = data_set[0:train_num]
    test_set = data_set[train_num:]
    return train_set, test_set, label_list


# 计算信息熵
# [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no']]
def calc_entropy(data_set):
    class_count = {}
    data_len = len(data_set)
    for sample in data_set:
        temp_class = sample[-1]
        if temp_class not in class_count:
            class_count[temp_class] = 0
        
        class_count[temp_class] += 1
    
    entropy = 0.0
    for key in class_count:
        prob = float(class_count[key]) / data_len
        entropy -= prob * log(prob, 2)
    
    return entropy

def split_dataset(train_set, test_set, feature_no, feature_val):
    split_train = []
    for sample in train_set:
        # 去除对应的feature_val那列，剩下的再组装返回
        if feature_val == sample[feature_no]:
            temp_sample = sample[:feature_no]
            temp_sample.extend(sample[feature_no + 1:])
            split_train.append(temp_sample)

    split_test = []
    for sample in test_set:
        # 去除对应的feature_val那列，剩下的再组装返回
        if feature_val == sample[feature_no]:
            temp_sample = sample[:feature_no]
            temp_sample.extend(sample[feature_no + 1:])
            split_test.append(temp_sample)
    
    return split_train, split_test

def split_trainset(train_set, feature_no, feature_val):
    split_train = []
    for sample in train_set:
        # 去除对应的feature_val那列，剩下的再组装返回
        if feature_val == sample[feature_no]:
            temp_sample = sample[:feature_no]
            temp_sample.extend(sample[feature_no + 1:])
            split_train.append(temp_sample)
    
    return split_train

def choose_best_feature(data_set):
    feature_num  = len(data_set[0]) - 1
    base_entropy = calc_entropy(data_set)
    gain_list = []
    # best_gain = 0.0
    # best_feature_no = 0
    for feature_no in range(feature_num):
        # 列表推导，获得唯一的特征值
        feature_val_list = [sample[feature_no] for sample in data_set]
        feature_val_set = set(feature_val_list)
        temp_entropy = 0.0
        temp_iv = 0.0
        for feature_val in feature_val_set:
            # 根据每一个特征值分裂数据求信息熵
            new_dataset = split_trainset(data_set, feature_no, feature_val)
            temp_prob = float(len(new_dataset)) / float(len(data_set))
            temp_entropy += temp_prob * calc_entropy(new_dataset)
            temp_iv += - temp_prob * log(temp_prob, 2)
    
        # 根据此特征划分产生的信息增益,信息增益率
        # 先取信息增益大于平均值，再取最大增益率
        temp_gain = base_entropy - temp_entropy
        if temp_iv != 0:
            temp_gain_rate = base_entropy / temp_iv
        else:
            temp_gain_rate = 0.0
        print("For Feature No:", feature_no, " Its Gain Is:", temp_gain, " Gain Rate Is:", temp_gain_rate)
        gain_list.append([feature_no, temp_gain, temp_gain_rate])
        '''
        if (temp_gain > best_gain):
            best_gain = temp_gain
            best_feature_no = feature_no
        '''
    # 按列相加
    gain_sum_list = np.sum(gain_list, axis=0)
    gain_sum = gain_sum_list[1]
    gain_average = gain_sum / feature_num
    print("Gain Sum:", gain_sum, " Gain Average:", gain_average)
    # 列表生成直接取大于平均值的增益
    select_gain_list = [gain for gain in gain_list if gain[1] > gain_average]
    # 按照第2维即增益率排序
    sorted_gain_list = sorted(select_gain_list, key=operator.itemgetter(2), reverse=True)
    print("Sorted Gain List:", sorted_gain_list)
    return sorted_gain_list[0][0]         

def get_max_class(class_list):
    class_count = {}
    for temp_class in class_list:
        if temp_class not in class_count.keys():
            class_count[temp_class] = 0

        class_count[temp_class] += 1

    # 按照第1维的数据排序，表里对应的是value，key是第0维
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]

# 根据特征，找每个特征值对应的最大类别
# {0:'Yes', 1:'Yes', 2:'No'}
def get_feature_max_class(test_set, best_feature_no):
    feature_val_list = [sample[best_feature_no] for sample in test_set]
    unique_feature_val_list = list(set(feature_val_list))

    feature_val_count = {}
    for i in range(len(unique_feature_val_list)):
        class_list = [sample[-1] for sample in test_set if sample[best_feature_no] == unique_feature_val_list[i]]
        max_class  = get_max_class(class_list)
        feature_val_count[unique_feature_val_list[i]] = max_class

    return feature_val_count

# 是否需要预剪枝，降低过拟合
def need_pre_cut(train_set, test_set, best_feature_no):
    # 划分前的分类正确率
    class_list  = [sample[-1] for sample in train_set]
    max_class   = get_max_class(class_list)
    correct_set = [sample for sample in test_set if max_class == sample[-1]]
    before_divide_prob = len(correct_set) / len(test_set)

    # 划分后的分类正确率
    # 随机划分训练集和测试集会出现：某个特征值训练集中不存在
    feature_val_count = get_feature_max_class(train_set, best_feature_no)
    correct_set = [sample for sample in test_set if feature_val_count[sample[best_feature_no]] == sample[-1]]
    after_divide_prob = len(correct_set) / len(test_set)

    return after_divide_prob > before_divide_prob


def create_tree(train_set, test_set, label_list):
    class_list = [sample[-1] for sample in train_set]
    # 所有class类别都相同，不用再拆分
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    
    # 只剩一个Yes/No，划分完成
    if 1 == len(train_set[0]):
        return get_max_class(class_list)

    best_feature_no = choose_best_feature(train_set)
    best_feature_label = label_list[best_feature_no]
    # 是否需要预剪枝
    '''
    if not need_pre_cut(train_set, test_set, best_feature_no):
        print("For Feature No:", best_feature_no, " Need Pre Cut")
        return get_max_class(class_list)
    '''

    decision_tree = { best_feature_label : {} }
    del(label_list[best_feature_no])

    feature_val_list = [sample[best_feature_no] for sample in train_set]
    unique_feature_val = set(feature_val_list)
    for val in unique_feature_val:
        temp_train_set, temp_test_set = split_dataset(train_set, test_set, best_feature_no, val)
        # 传参时列表是引用，可能会提前删除某个特征
        temp_label_list = label_list[:]
        decision_tree[best_feature_label][val] = create_tree(temp_train_set, temp_test_set, temp_label_list)
    
    return decision_tree


train_set, test_set, label_list = read_data('fruit_data.txt')
print('Train:', train_set, 'Test:', test_set, ' Label:', label_list)

# data_set = [[0, 1, 1, 'yes'], [1, 1, 1, 'yes'], [2, 1, 0, 'no'], [3, 0, 0, 'no'], [4, 0, 1, 'no']]
# label_list = ['No', 'First', 'Second']
tree = create_tree(train_set, test_set, label_list)
print(tree)
'''
entropy = calc_entropy(data_set)
print(entropy)
data_set = split_dataset(data_set, 1, 0)
print(data_set)
'''
# feature_no = choose_best_feature(data_set)
# print(feature_no)