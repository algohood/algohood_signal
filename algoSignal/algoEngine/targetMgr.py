# flake8: noqa
"""
@Create: 2024/9/30 14:06
@File: targetMgr.py
@Author: Jijingyuan
"""
import importlib
import uuid
import numpy as np
from algoUtils.reloadUtil import reload_all
from algoUtils.baseUtil import TargetBase, Sample
from algoUtils.schemaUtil import TargetMgrParam
from ..algoConfig.profileConfig import profile_stats
from typing import Optional, Dict, List, Union 


class TargetMgr:

    def __init__(self, _target_info: Optional[TargetMgrParam]):
        self.target_info = _target_info
        self.fields = None
        self.samples = {}

    def get_target_method(self, _method_info: Optional[TargetMgrParam]) -> Optional[TargetBase]:
        if _method_info is None:
            return

        self.fields = [_method_info.target_fields] if isinstance(_method_info.target_fields, str) else _method_info.target_fields
        module = importlib.import_module('algoStrategy.algoTargets.{}'.format(_method_info.target_method_name))
        reload_all(module)
        cls_method = getattr(module, 'Algo')
        if cls_method is None:
            raise Exception('Unknown Method: {}'.format(_method_info.target_method_name))
        
        instance = cls_method(**_method_info.target_method_param)
        return instance
    
    @profile_stats.update_cost_time
    def add_sample(self, _sample: Sample):
        target_mgr = self.get_target_method(self.target_info)
        if target_mgr is None:
            return
        
        target_mgr.init_instance(_sample.signal)
        _sample.fields = self.fields
        if _sample.sim_targets:
            intercept_targets = {field: _sample.sim_targets[field] for field in self.fields} if self.fields else _sample.sim_targets
            _sample.sim_intercept = target_mgr.intercept_signal_given_targets(intercept_targets)

        self.samples[str(uuid.uuid4())] = [target_mgr, _sample]

    @profile_stats.update_cost_time
    def get_completed_samples(self, _current_ts: float, _data: Dict[str, np.ndarray]) -> list[Sample]:
        completed_samples = []
        sample_ids = list(self.samples.keys())
        for sample_id in sample_ids:
            [target_mgr, sample] = self.samples[sample_id]
            targets = target_mgr.generate_targets(_current_ts, _data)
            if targets is not None:
                sample.targets = targets
                intercept_targets = {field: targets[field] for field in self.fields} if self.fields else targets
                sample.intercept = target_mgr.intercept_signal_given_targets(intercept_targets)
                completed_samples.append(sample)
                del self.samples[sample_id]

        return completed_samples
    