# flake8: noqa
"""
@Create: 2024/9/30 14:06
@File: signalMgr.py
@Author: Jijingyuan
"""
import importlib
import numpy as np
from typing import Optional, Dict

from algoUtils.reloadUtil import reload_all
from algoUtils.baseUtil import SignalBase
from algoUtils.baseUtil import Signal
from algoUtils.schemaUtil import SignalMgrParam
from ..algoConfig.profileConfig import profile_stats


class SignalMgr:

    def __init__(self, _signal_info: SignalMgrParam):
        self.signal_mgr = self.get_signal_method(_signal_info)
        self.cool_down_ts = _signal_info.cool_down_ts
        self.signal_cache = {}


    @staticmethod
    def get_signal_method(_method_info: SignalMgrParam) -> SignalBase:
        module = importlib.import_module('algoStrategy.algoSignals.{}'.format(_method_info.signal_method_name))
        reload_all(module)
        cls_method = getattr(module, 'Algo')
        if cls_method is None:
            raise Exception('Unknown Method: {}'.format(_method_info.signal_method_name))
        
        instance = cls_method(**_method_info.signal_method_param)
        return instance

    @profile_stats.update_cost_time
    def update_signals(self, _current_ts: float, _data: Dict[str, np.ndarray]) -> Optional[Signal]:
        self.signal_mgr.update_state(_current_ts, _data)
        signal = self.signal_mgr.generate_signals(_current_ts, _data)
        if signal is None:
            return

        last_ts = self.signal_cache.get(signal.symbol)
        if last_ts is None:
            signal.timestamp = _current_ts
            self.signal_cache[signal.symbol] = _current_ts
            return signal

        delta_ts = _current_ts - last_ts
        if delta_ts > self.cool_down_ts:
            signal.timestamp = _current_ts
            self.signal_cache[signal.symbol] = _current_ts
            return signal
