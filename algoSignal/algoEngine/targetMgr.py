# flake8: noqa
"""
@Create: 2024/9/30 14:06
@File: targetMgr.py
@Author: Jijingyuan
"""
import importlib

from algoUtils.reloadUtil import reload_all
from algoUtils.baseUtil import TargetBase
from algoUtils.baseUtil import Sample
from algoUtils.schemaUtil import TargetMgrParam
from typing import Optional


class TargetMgr:

    def __init__(self, _target_info: Optional[TargetMgrParam]):
        self.target_mgr = self.get_target_method(_target_info)

    @staticmethod
    def get_target_method(_method_info: Optional[TargetMgrParam]) -> TargetBase:
        if _method_info is None:
            return

        module = importlib.import_module('algoStrategy.algoTargets.{}'.format(_method_info.target_method_name))
        reload_all(module)
        cls_method = getattr(module, 'Algo')
        if cls_method is None:
            raise Exception('Unknown Method: {}'.format(_method_info.target_method_name))
        
        instance = cls_method(**_method_info.target_method_param)
        return instance
    
    def add_sample(self, _sample: Sample):
        if self.target_mgr is None:
            return
        
        self.target_mgr.add_sample(_sample)

    def get_completed_samples(self, _data) -> list[Sample]:
        if self.target_mgr is None:
            return []
        
        completed_samples = self.target_mgr.update_targets(_data) or []
        for completed_sample in completed_samples:
            if completed_sample.targets is None:
                raise ValueError('targets is None')
            
            if completed_sample.features is None:
                raise ValueError('features is None')
            
            if completed_sample.signal is None:
                raise ValueError('signal is None')
            
            if completed_sample.forecast is None:
                raise ValueError('forecast is None')
            
            if completed_sample.actual is None:
                raise ValueError('actual is None')

        return completed_samples
