# safe_app.py
import asyncio
import os
import sys

# 1. 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:False'
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
os.environ['PYTORCH_JIT'] = '0'
# 新增：禁用Streamlit文件监控和XSRF保护
os.environ["STREAMLIT_SERVER_WATCH_MODULES"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION"] = "false"
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_ENABLE_CORS"] = "false"

# 设置事件循环策略（解决asyncio问题）
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
else:
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
# 2. 设置 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
simulate_dir = os.path.join(script_dir, '模拟')

# 将模拟目录添加到 sys.path
sys.path.insert(0, simulate_dir)

# 3. 导入并修改 torch
import torch
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.set_default_tensor_type(torch.FloatTensor)

# 4. 导入原始应用所需的其他模块
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('agg')

# 5. 提前导入您的自定义模块
from vae_gen_1_随机生成属性组合 import P_VariationalAutoencoder
from vae_gen_2_随机生成成分数据 import C_VariationalAutoencoder, normalize_filter, normalize_top_n_values
from vae_gen_3_网格过滤 import voxel_filter
from vae_gen_4_机器学习模型排序 import ML_sort
from vae_gen_5_将排序后的成分转化为化学式 import Expression_formula

# 6. 读取并执行原始应用
original_app_path = os.path.join(simulate_dir, 'streamlit_app.py')
with open(original_app_path, 'r', encoding='utf-8') as f:
    app_code = f.read()

# 创建包含所有必要模块的全局命名空间
exec_namespace = {
    '__file__': original_app_path,
    '__name__': '__main__',
    'os': os,
    'sys': sys,
    're': re,
    'np': np,
    'pd': pd,
    'st': st,
    'torch': torch,
    'matplotlib': matplotlib,
    'P_VariationalAutoencoder': P_VariationalAutoencoder,
    'C_VariationalAutoencoder': C_VariationalAutoencoder,
    'normalize_filter': normalize_filter,
    'normalize_top_n_values': normalize_top_n_values,
    'voxel_filter': voxel_filter,
    'ML_sort': ML_sort,
    'Expression_formula': Expression_formula,
}

# 执行代码
exec(app_code, exec_namespace)