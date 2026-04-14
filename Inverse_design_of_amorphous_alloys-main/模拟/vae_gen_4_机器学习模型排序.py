import os

import numpy as np

import joblib


def ML_sort(Generated_com, Bs, Hc, Tg):
    # 获取当前脚本文件的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))


    # 构建模型文件的绝对路径
    BS_model = os.path.join(script_dir, '生成的模型', 'TL_model.pkl')
    HC_model = os.path.join(script_dir, '生成的模型', 'TS_model.pkl')
    TG_model = os.path.join(script_dir, '生成的模型', 'σb_model.pkl')

    model_BS = joblib.load(BS_model)
    model_HC = joblib.load(HC_model)
    model_TG = joblib.load(TG_model)

    BS_pred = model_BS.predict(Generated_com)
    HC_pred = model_HC.predict(Generated_com)
    TG_pred = model_TG.predict(Generated_com)

    pred_att = np.concatenate([BS_pred.reshape(-1, 1), HC_pred.reshape(-1, 1), TG_pred.reshape(-1, 1)],axis=1)

    label =np.array([Bs, Hc, Tg])
    num_label  = np.tile(label, (Generated_com.shape[0], 1))

    # mse = np.sum(np.array([0.1, 0.1, 1])*(pred_att - num_label)**2,axis=1)

    # r2 = calculate_multiple_r2(num_label, pred_att)

    a = num_label
    b = pred_att

    # 计算每行向量的范数（L2 范数）
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)

    # 计算每对行向量的内积
    dot_product = np.sum(a * b, axis=1)

    # 计算余弦相似度
    cosine_similarity = dot_product / (norm_a * norm_b)


    # Generated_com_['Bs'] = BS_pred
    # Generated_com_['Hc'] = HC_pred
    # Generated_com_['Tg'] = TG_pred
    # Generated_com['Property_cosine_similarity '] =  cosine_similarity
    # Generated_com_sorted = Generated_com.sort_values(by='Property_cosine_similarity ')
    # mse =np.array(Generated_com_sorted.iloc[:, -1])
    # Generated_com_sorted = Generated_com_sorted.drop(Generated_com_sorted.columns[-1], axis=1)
    #
    return  cosine_similarity


# Generated_com_sorted.to_excel('./生成数据/Generated_component_sort.xlsx', index=False)
#
# print('排序完成')
