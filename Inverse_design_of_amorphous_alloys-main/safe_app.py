# safe_app.py
import os
import sys

# ====== 您提供的代码放在这里开始 ======
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 明确告知系统无可用GPU
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:False'
os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
os.environ['PYTORCH_JIT'] = '0'  # 禁用JIT编译，减少CUDA相关调用
# ====== 您提供的代码放在这里结束 ======

# 紧接着，导入“阉割”过的torch
import torch
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
torch.set_default_tensor_type(torch.FloatTensor)

# 然后，导入并运行您原来的应用文件
original_app_path = os.path.join(os.path.dirname(__file__), '模拟', 'streamlit_app.py')
with open(original_app_path, 'r', encoding='utf-8') as f:
    exec(f.read(), {'__file__': original_app_path, '__name__': '__main__', 'torch': torch})