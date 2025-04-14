# flake8: noqa
"""
@Create: 2024/9/30 14:06
@File: interceptMgr.py
@Author: Jijingyuan
"""
import importlib
from algoUtils.reloadUtil import reload_all
from algoUtils.baseUtil import ModelBase, Sample, SelectorBase
from algoUtils.schemaUtil import ModelMgrParam
from typing import Optional, Dict, List
from ..algoConfig.profileConfig import profile_stats


class ModelMgr:

    def __init__(self, _model_info: Optional[ModelMgrParam]):
        self.model_mgr = self.get_model_method(_model_info)
        self.selector_mgr = self.get_selector_method(_model_info)
        self.cache_size = _model_info.cache_size if _model_info else 100
        self.retain_size = _model_info.retain_size if _model_info else 50
        self.cache = []

    @staticmethod
    def get_model_method(_method_info: Optional[ModelMgrParam]) -> Optional[ModelBase]:
        if _method_info is None:
            return

        module = importlib.import_module('algoStrategy.algoModels.{}'.format(_method_info.model_method_name))
        reload_all(module)
        cls_method = getattr(module, 'Algo')
        if cls_method is None:
            raise Exception('Unknown Method: {}'.format(_method_info.model_method_name))
        
        instance = cls_method(**_method_info.model_method_param)
        return instance

    @staticmethod
    def get_selector_method(_method_info: Optional[ModelMgrParam]) -> Optional[SelectorBase]:
        if _method_info is None:
            return

        module = importlib.import_module('algoStrategy.algoModels.{}'.format(_method_info.selector_method_name))
        reload_all(module)
        cls_method = getattr(module, 'Algo')
        if cls_method is None:
            raise Exception('Unknown Method: {}'.format(_method_info.selector_method_name))
        
        instance = cls_method(**_method_info.selector_method_param)
        return instance


    @profile_stats.update_cost_time
    def predict_target(self, _features: Dict[str, float]) -> Optional[Dict[str, float]]:
        if self.model_mgr is None or not _features:
            return
        
        return self.model_mgr.predict(_features)
    
    @profile_stats.update_cost_time
    def update_model(self, _samples: List[Sample]):
        if not self.model_mgr or not self.selector_mgr:
            return
        
        features = []
        targets = []
        self.cache.extend(_samples)
        if len(self.cache) >= self.cache_size:
            for sample in self.cache:
                if sample.features:
                    features.append(sample.features)

                if sample.targets:
                    if sample.fields:
                        targets.append({field: sample.targets[field] for field in sample.fields})
                    else:
                        targets.append(sample.targets)
            
            feature_fields = self.selector_mgr.select_features(features, targets)
            if feature_fields:
                features = [{k: v for k, v in feature.items() if k in feature_fields} for feature in features]

            if len(features) == len(targets):
                self.model_mgr.train_model(features, targets)
                self.cache = self.cache[-self.retain_size:]
            else:
                self.cache = self.cache[-self.cache_size:]
