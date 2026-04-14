import os
import re
import streamlit as st
import numpy as np
import pandas as pd
import torch
from vae_gen_1_随机生成属性组合 import P_VariationalAutoencoder
from vae_gen_2_随机生成成分数据 import C_VariationalAutoencoder, normalize_filter, normalize_top_n_values
from vae_gen_3_网格过滤 import voxel_filter
from vae_gen_4_机器学习模型排序 import ML_sort
from vae_gen_5_将排序后的成分转化为化学式 import Expression_formula
import matplotlib

matplotlib.use('agg')

# 获取当前脚本文件的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))


p_vae = P_VariationalAutoencoder(3)
c_vae = C_VariationalAutoencoder(3)
# 构建模型文件的绝对路径
p_model_path = os.path.join(script_dir, 'newtest_model', '属性模型新.pth')
c_model_path = os.path.join(script_dir, 'newtest_model', '成分模型有权重（d=3）(50,0.1,0.01).pth')


p_vae.load_state_dict(torch.load(p_model_path, map_location=torch.device('cpu')))


c_vae.load_state_dict(torch.load(c_model_path,map_location=torch.device('cpu') ))

def Generated_Attribute(num, Re_ts, Re_tl, Re_σb):
    sample = torch.normal(0, 1, (num, 3))

    # tensor_sam = torch.tensor(sample).float().to(device)
    attribute_data = p_vae.decoder(sample)

    re_attribute_mse = torch.sum((attribute_data - p_vae(attribute_data.float())) ** 2, dim=1)

    attribute_df = pd.DataFrame(attribute_data.cpu().detach().numpy(), columns=['TS', 'TL', 'σb'])
    mse = re_attribute_mse.cpu().detach().numpy()

    attribute_df['Re_attribute_mse'] = mse

    df_filter = attribute_df[(attribute_df['TS'].between(Re_ts*0.1, np.inf)) &
                             ((attribute_df['TL']).between(Re_tl*0.1, np.inf)) &
                             ((attribute_df['σb']).between( Re_σb*0.5, np.inf)) &
                             ((attribute_df['Re_attribute_mse']).between(0, 0.003))
                             ]

    df_filter.iloc[:, 0] = df_filter.iloc[:, 0] * 10
    df_filter.iloc[:, 1] = df_filter.iloc[:, 1] * 10
    df_filter.iloc[:, 2] = df_filter.iloc[:, 2] * 2

    # 为不同列设置不同的格式
    format_dict = {
        'TS': "{:,.4f}",
        'TL': "{:,.4f}",
        'σb': "{:,.4f}"
    }
    df_filter1 = df_filter.style.format(format_dict)


    return 'Initially define the amount to be generated: ' + str(num) + \
           '\n' + 'The final quantity generated: '+str(df_filter.shape[0]), df_filter1

def Generated_Component(TS, TL, σb, Gen_num, Keep_num, Voxel_size):
    TS=TS/10
    TL = TL / 10
    σb = σb / 2
    attribute_data = np.array([TS, TL, σb])
    attribute_data = torch.tensor(attribute_data)
    re_attribute_data = p_vae(attribute_data.float())
    re_attribute_rate = ((attribute_data - re_attribute_data) ** 2).sum()

    if re_attribute_rate < 0.003:

        sample = torch.normal(0, 1, (Gen_num, 3))
        tensor_repeated = re_attribute_data.repeat(Gen_num, 1)
        re_c = c_vae.decoder(sample, tensor_repeated.float())

        if Keep_num:
            filter_c = normalize_top_n_values(re_c, Keep_num, threshold=0.0001)
        else:
            filter_c = normalize_filter(re_c, threshold=0.01)

        filter_c_np = filter_c.cpu().detach().numpy()
        # filter_c_np = filter_c_np * 100
        filter_data = voxel_filter(np.array(filter_c_np), Voxel_size)

        if Voxel_size == 0:
            voxel_out = 'No filter' + '\n' + 'Final generated quantity: ' + str(filter_data.shape[0])

        else:
            voxel_out = 'Filter pixel size is: ' + str(Voxel_size) + '\n' + 'The final quantity generated: ' + str(
                filter_data.shape[0])

        # 机器学习模型排序
        ml_cosine_similarity = ML_sort(filter_data, TS, TL, σb)

        # 表示为化学式
        formula_list, non_zero_num_list = Expression_formula(filter_data)

        formula_df = pd.DataFrame(formula_list, columns=['Generated result'])

        formula_df['Element quantity'] = non_zero_num_list
        formula_df['Possibility ranking'] = ml_cosine_similarity

        formula_df = formula_df.sort_values(by='Possibility ranking', ascending=False)

        # formula_df = formula_df.head(100)

        return 'The rate of performance rebuild must be less than  ' + str(re_attribute_rate.item()) + \
               '\n' + 'Initially define the amount to be generated: ' + str(Gen_num) + \
               '\n' + voxel_out, formula_df

    else:
        return 'Performance reconstruction rate: ' + str(
            re_attribute_rate.item()) + ', there may be a problem with the input attribute composition, please check.', None


def Select_Component(formula_df, select_elem):
    # 构建正则表达式模式
    pattern = ''.join(f'(?=.*{re.escape(elem)})' for elem in select_elem)

    # 使用正则表达式进行筛选
    select_elem_df = formula_df[formula_df['Generated result'].str.contains(pattern, regex=True)]

    # 检查筛选后的数据是否为空
    if select_elem_df.empty:
        voxel_out = "No data containing this element was found. You can try to increase the number of alloy components generated or limit the number of elements used. "
        formula_df = pd.DataFrame(select_elem_df, columns=['Generated result'])
        formula_df['Element quantity'] = 0
        formula_df['Possibility ranking'] = 0

    else:
        voxel_out = 'The number of data containing the required elements is: ' + str(select_elem_df.shape[0])

    return 'The required elements are: ' + str(select_elem) + \
           '\n' + voxel_out, select_elem_df


# Streamlit 应用界面
st.title("Inverse design of Sn-based solder via generative AI: from multi-objective properties to tailored compositions")

# 生成属性组合选项卡
st.markdown("## Generate Property Combination")
st.markdown('By leveraging the internal connections of Solidus Temperature (TS), Liquidus Temperature (TL), and Tensile Strength (σb), possible combinations of the three properties are generated.')

# 初始化 session_state
if 'num_generated_output' not in st.session_state:
    st.session_state.num_generated_output = None
if 'data_output' not in st.session_state:
    st.session_state.data_output = None
if 'result_text' not in st.session_state:
    st.session_state.result_text = None
if 'result_dataframe1' not in st.session_state:
    st.session_state.result_dataframe1 = None
if 'result_text2' not in st.session_state:
    st.session_state.result_text2 = None
if 'result_dataframe2' not in st.session_state:
    st.session_state.result_dataframe2 = None

col1 =  st.columns(1)[0]
col2, col3 = st.columns(2)
col4 =  st.columns(1)[0]

with col1:
    num_combinations_input = st.number_input("Enter the number of combinations of generating properties (TS, TL, σb). ", value=0)

with col2:

    bs_lower_input = st.number_input("Lower bound of TS: ", value=217.0, step=0.1, min_value=90.0)
    hc_lower_input = st.number_input("Lower bound of TL: ", value=217.0, step=0.1, min_value=90.0)


with col3:
    tg_lower_input = st.number_input("Lower bound of σb: ", value=30.0, step=0.1,
                                     min_value=13.8)
    generate_property_button = st.button("Generate Property Combination")
    if generate_property_button:
        num_generated_output, data_output = Generated_Attribute(num_combinations_input, bs_lower_input, hc_lower_input,
                                                                tg_lower_input)
        st.session_state.num_generated_output = num_generated_output
        st.session_state.data_output = data_output


with col4:
    if st.session_state.num_generated_output is not None:
        st.subheader("Property generation information.")
        st.text(st.session_state.num_generated_output)
        st.subheader("Randomly generated combinations of properties.")
        st.dataframe(st.session_state.data_output)



# 生成成分数据选项卡
st.markdown("## Generate Component Data")
st.markdown('Potential alloy components are generated using the properties of TS, TL, and σb.')

col3, col4 = st.columns(2)

with col3:
    Bs_input = st.number_input("TS", step=0.1, min_value=90.0, value=None)
    Hc_input = st.number_input("TL", step=0.1, min_value=90.0, value=None)
    Tg_input = st.number_input("σb", step=0.01, min_value=13.8, value=None)

with col4:
    num_generated_components_input = st.number_input("Generated component quantity: More components increase the probability of achieving desired results.", value=0)
    max_elements_input = st.number_input(
        'Specify max element count. Randomly use a varying number of elements when unspecified.',
        step=1, max_value=17, value=0)
    grid_filter_size_input = st.number_input('Grid filter size to remove similar components', step=0.01, value=0.0)

col5 = st.columns(1)[0]

with col5:
    generate_component_button = st.button("Generate Component Data")
    required_elements_input = st.multiselect(
        'Filter for required elements: Filter out components with the required elements from randomly generated data.',
        ['Ag', 'Cu', 'Zn', 'Cd', 'In', 'Sn', 'Ga', 'Bi', 'Ni', 'Ti', 'Sb', 'Dy', 'Mn', 'Al', 'Y', 'Ce',
                       'Co'])
    select_button = st.button("Select element ")

    if generate_component_button:
        result_text, result_dataframe1 = Generated_Component(
            Bs_input, Hc_input, Tg_input, num_generated_components_input,
            max_elements_input, grid_filter_size_input
        )
        st.session_state.result_text = result_text
        st.session_state.result_dataframe1 = result_dataframe1

    if st.session_state.result_text is not None:
        st.subheader("Components generate information")
        st.text(st.session_state.result_text)
        st.subheader("Composition data designed according to properties combination")
        st.dataframe(st.session_state.result_dataframe1)

    if select_button:
        if st.session_state.result_dataframe1 is None:
            st.error("The components have not been generated！")
        else:
            result_text2, result_dataframe2 = Select_Component(
                st.session_state.result_dataframe1, required_elements_input
            )
            st.session_state.result_text2 = result_text2
            st.session_state.result_dataframe2 = result_dataframe2

    if st.session_state.result_text2 is not None:
        st.subheader("Components filtering information")
        st.text(st.session_state.result_text2)
        st.subheader("Contains data for the selected element")
        st.dataframe(st.session_state.result_dataframe2)
