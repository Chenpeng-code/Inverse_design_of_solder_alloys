import numpy as np
import pandas as pd

def Expression_formula(gen_data):
    """将生成的数据转换为化学式（使用LaTeX格式）"""
    column_names = ['Ag', 'Cu', 'Zn', 'Cd', 'In', 'Sn', 'Ga', 'Bi', 'Ni', 'Ti', 'Sb', 'Dy', 'Mn', 'Al', 'Y', 'Ce', 'Co']

    df = pd.DataFrame(gen_data, columns=column_names)

    formula_list = []
    non_zero_num_list = []

    for i in range(df.shape[0]):
        zero_indices = np.where(df.iloc[i] != 0)[0]

        num_non_zero = np.array(np.where(df.iloc[i] != 0)).shape[1]
        non_zero_num_list.append(num_non_zero)

        relevant_data = df.iloc[i].iloc[zero_indices]
        # 按照数字从大到小排序
        relevant_data_sorted = relevant_data.sort_values(ascending=False)
        # 生成列名与数字结合成的字符串
        chemical_formula = ''
        for idx in relevant_data_sorted .index:
            element = idx
            amount = relevant_data_sorted [idx] * 100
            if amount == 1:
                chemical_formula += element
            else:
                chemical_formula += f"{element}{amount:.2f}"  # 调整格式，例如保留小数点后三位

        formula_list.append(chemical_formula)
        # print(chemical_formula)

    return formula_list, non_zero_num_list