# flake8: noqa
"""
@Create: 2024/9/30 14:06
@File: targetMgr.py
@Author: Jijingyuan
"""
import importlib
import uuid
from collections import deque

from algoUtils.reloadUtil import reload_all
from algoUtils.baseUtil import TargetBase
from algoUtils.baseUtil import Sample
from algoUtils.schemaUtil import TargetMgrParam, PerformanceMgrParam
from algoSignal.algoEngine.performanceMgr import PerformanceMgr
from typing import Optional, Dict, List


class TargetMgr:

    def __init__(self, _performance_info: Optional[PerformanceMgrParam], _target_info: Optional[TargetMgrParam]):
        self.performance_info = _performance_info
        self.target_info = _target_info
        self.samples = {}

    @staticmethod
    def get_target_method(_method_info: Optional[TargetMgrParam]) -> TargetBase:
        if _method_info is None:
            return
        
        if _method_info.use_performance is True:
            return

        module = importlib.import_module('algoStrategy.algoTargets.{}'.format(_method_info.target_method_name))
        reload_all(module)
        cls_method = getattr(module, 'Algo')
        if cls_method is None:
            raise Exception('Unknown Method: {}'.format(_method_info.target_method_name))
        
        instance = cls_method(**_method_info.target_method_param)
        return instance
    
    @staticmethod
    def get_performance_method(_method_info: Optional[PerformanceMgrParam]):
        if _method_info is None:
            return
        
        return PerformanceMgr(_method_info)
    
    def init_target_instance(self, _sample: Sample):
        target_mgr = self.get_target_method(self.target_info)
        if target_mgr is None:
            return
        
        target_mgr.init_instance(_sample.signal)
        if _sample.sim_targets is not None and self.target_info.use_performance is False:
            _sample.sim_intercept = target_mgr.intercept_signal_given_targets(_sample.sim_targets)
        return target_mgr
    
    def init_performance_instance(self, _sample: Sample):
        performance_mgr = self.get_performance_method(self.performance_info)
        if performance_mgr is None:
            return
        
        performance_mgr.performance_mgr.init_instance(_sample.signal)
        if self.target_info is not None:
            if _sample.sim_targets is not None and self.target_info.use_performance is True:
                _sample.sim_intercept = True if _sample.sim_targets['win'] < 1 else False
        return performance_mgr

    def add_sample(self, _sample: Sample):
        target_mgr = self.init_target_instance(_sample)
        performance_mgr = self.init_performance_instance(_sample)
        if target_mgr is None and performance_mgr is None:
            return
        
        sample_id = str(uuid.uuid4())
        self.samples[sample_id] = [target_mgr, performance_mgr, _sample]

    def get_completed_samples(self, _data: Dict[str, List[List]]) -> list[Sample]:
        completed_samples = []
        sample_ids = list(self.samples.keys())
        for sample_id in sample_ids:
            [target_mgr, performance_mgr, sample] = self.samples[sample_id]
            if performance_mgr is not None:
                performance = performance_mgr.update_performance(_data)
                if performance is not None:
                    sample.performance = performance
                    performance_mgr = None
                    
                    # 如果启用了performance且performance成功，则更新样本数据
                    if self.target_info is not None and performance.success and self.target_info.use_performance is True:
                        sample.targets = {'win': performance.is_win}
                        sample.intercept = False if performance.is_win > 0 else True
                        completed_samples.append(sample)
            
            # 处理目标管理器
            if target_mgr is not None:
                targets = target_mgr.generate_targets(_data)
                if targets is not None:
                    sample.targets = targets
                    sample.intercept = target_mgr.intercept_signal_given_targets(targets)
                    completed_samples.append(sample)
                    target_mgr = None

            if target_mgr is None and performance_mgr is None:
                del self.samples[sample_id]

        return completed_samples
    