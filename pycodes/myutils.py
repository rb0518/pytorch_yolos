# 2025-10-29 通过导出关键Tensor数据，辅助C++程序判定那一步输入输出数据异常
"""
    Pytorch与Libtorch中同名函数功能存在差异torch.save和torch::save是不相同的
    Pytorch torch.save导出的Tensor, 在Libtorch中要用pickle_load(vector<char>)的模式导入
    Libtorch torch::save是通过serialize导出, Pytorch要用torch::jit::load方式读取
"""
import torch
import io

def export_test_tensor_tofile(filename, x):
    xi = x.clone()
    f = io.BytesIO()
    torch.save(xi, f)
    
    with open(filename+'.pt', 'wb') as out_f:
        out_f.write(f.getbuffer())   

class GlobalDebugManger:
    """使用全局变量由该类管理"""
    _shared_state_ = {  "run_debug" : False ,
                        "start_record" : False , 
                        "one_time" : True }
    
    def __init__(self):
        self.__dict__ = self._shared_state_

    def open_debug():
        self._shared_state_["run_debug"] = True

    def close_debug():
        self._shared_state_["run_debug"] = False

    def is_debug():
        return self._shared_state_["run_debug"]

    def modify(self, key, value):
        self._shared_state_[key] = value

    def get_start_record_state(self):
        return self._shared_state_["start_record"]
    
    def set_start_record_state(self):
        self._shared_state_["start_record"] = True

    def get_one_time_state(self):
        return self._shared_state_["one_time"]
    
    def modify_one_time_state(self, new_state):
        self._shared_state_["one_time"] = new_state

