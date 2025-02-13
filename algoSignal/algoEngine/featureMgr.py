# flake8: noqa
"""
@Create: 2024/9/30 14:06
@File: featureMgr.py
@Author: Jijingyuan
"""
import importlib
from typing import Optional, List, Union
from algoUtils.reloadUtil import reload_all
from algoUtils.baseUtil import SignalBase
from algoUtils.schemaUtil import FeatureMgrParam

class FeatureMgr:

    def __init__(self, _feature_info: Optional[Union[List[FeatureMgrParam], FeatureMgrParam]]):
        self.feature_mgrs = self.init_feature_mgrs(_feature_info)
    
    def init_feature_mgrs(self, _feature_info: Optional[List[FeatureMgrParam]]):
        if _feature_info is None:
            return {}
        
        if isinstance(_feature_info, FeatureMgrParam):
            return {_feature_info.feature_method_name: self.get_feature_method(_feature_info)}
        
        return {v.feature_method_name: self.get_feature_method(v) for v in _feature_info} 
        
    @staticmethod
    def get_feature_method(_method_info: FeatureMgrParam) -> SignalBase:
        module = importlib.import_module('algoStrategy.algoFeatures.{}'.format(_method_info.feature_method_name))
        reload_all(module)
        cls_method = getattr(module, 'Algo')
        if cls_method is None:
            raise Exception('Unknown Method: {}'.format(_method_info.feature_method_name))
        
        instance = cls_method(**_method_info.feature_method_param)
        return instance
    
    def update_features(self, _data) -> dict:
        features = {}
        for feature_name, feature_mgr in self.feature_mgrs.items():
            if feature_mgr is None:
                continue
            
            feature_dict = feature_mgr.generate_features(_data) or {}
            features.update({'{}_{}'.format(feature_name, k): v for k, v in feature_dict.items()})
            
        return features
