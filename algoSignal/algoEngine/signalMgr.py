# flake8: noqa
"""
@Create: 2024/9/30 14:06
@File: signalMgr.py
@Author: Jijingyuan
"""
import importlib

from algoUtils.reloadUtil import reload_all
from algoUtils.baseUtil import SignalBase
from algoUtils.baseUtil import Signal
from algoUtils.schemaUtil import SignalMgrParam
from typing import Optional


class SignalMgr:

    def __init__(self, _signal_info: Optional[SignalMgrParam]):
        self.signal_mgr = self.get_signal_method(_signal_info)

    @staticmethod
    def get_signal_method(_method_info: Optional[SignalMgrParam]) -> SignalBase:
        if _method_info is None:
            return None
        
        module = importlib.import_module('algoStrategy.algoSignals.{}'.format(_method_info.signal_method_name))
        reload_all(module)
        cls_method = getattr(module, 'Algo')
        if cls_method is None:
            raise Exception('Unknown Method: {}'.format(_method_info.signal_method_name))
        
        instance = cls_method(**_method_info.signal_method_param)
        return instance
    
    def update_signals(self, _data) -> list[Signal]:
        return self.signal_mgr.generate_signals(_data) or []
        