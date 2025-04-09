# flake8: noqa
"""
@Create: 2024/9/30 14:06
@File: featureMgr.py
@Author: Jijingyuan
"""
import importlib
import numpy as np
from typing import Optional, List, Union, Dict

from algoUtils.reloadUtil import reload_all
from algoUtils.baseUtil import FeatureBase
from algoUtils.schemaUtil import FeatureMgrParam
from ..algoConfig.profileConfig import profile_stats


class FeatureMgr:

    def __init__(self, _feature_info: Optional[Union[List[FeatureMgrParam], FeatureMgrParam]]):
        self.feature_mgrs = self.init_feature_mgrs(_feature_info)
    
    def init_feature_mgrs(self, _feature_info: Optional[Union[List[FeatureMgrParam], FeatureMgrParam]]):
        if _feature_info is None:
            return {}
        
        if isinstance(_feature_info, FeatureMgrParam):
            return {_feature_info.feature_method_name: self.get_feature_method(_feature_info)}
        
        return {v.feature_method_name: self.get_feature_method(v) for v in _feature_info} 
        
    @staticmethod
    def get_feature_method(_method_info: FeatureMgrParam) -> FeatureBase:
        module = importlib.import_module('algoStrategy.algoFeatures.{}'.format(_method_info.feature_method_name))
        reload_all(module)
        cls_method = getattr(module, 'Algo')
        if cls_method is None:
            raise Exception('Unknown Method: {}'.format(_method_info.feature_method_name))
        
        instance = cls_method(**_method_info.feature_method_param)
        return instance
    
    def update_features(self, _current_ts: float, _data: Dict[str, np.ndarray]):
        for _, feature_mgr in self.feature_mgrs.items():
            if feature_mgr is None:
                continue
            
            feature_mgr.update_state(_current_ts, _data)

    def generate_features(self, _current_ts: float, _data: Dict[str, np.ndarray]):
        features = {}
        for _, feature_mgr in self.feature_mgrs.items():
            if feature_mgr is None:
                continue
            
            feature_result = feature_mgr.generate_features(_current_ts, _data)
            if isinstance(feature_result, dict):
                features.update(feature_result)

        return features
