# flake8: noqa
"""
@Create: 2024/9/30 14:06
@File: interceptMgr.py
@Author: Jijingyuan
"""
import importlib

from algoUtils.reloadUtil import reload_all
from algoUtils.baseUtil import InterceptBase
from algoUtils.schemaUtil import InterceptMgrParam
from typing import Optional


class InterceptMgr:

    def __init__(self, _intercept_info: Optional[InterceptMgrParam]):
        self.intercept_mgr = self.get_intercept_method(_intercept_info)

    @staticmethod
    def get_intercept_method(_method_info: Optional[InterceptMgrParam]) -> InterceptBase:
        if _method_info is None:
            return

        module = importlib.import_module('algoStrategy.algoIntercepts.{}'.format(_method_info.intercept_method_name))
        reload_all(module)
        cls_method = getattr(module, 'Algo')
        if cls_method is None:
            raise Exception('Unknown Method: {}'.format(_method_info.intercept_method_name))
        
        instance = cls_method(**_method_info.intercept_method_param)
        return instance

    def handle_forecast(self, _features):
        if self.intercept_mgr is None:
            return False
        
        target = self.intercept_mgr.forcast_target(_features)
        is_intercepted = self.intercept_mgr.check_target(target)
        return is_intercepted
    
    def handle_actual(self, _target):
        is_intercepted = self.check_target(_target)
        return is_intercepted

    def update_model(self, _features_and_targets):
        self.intercept_mgr.update_model(_features_and_targets)
