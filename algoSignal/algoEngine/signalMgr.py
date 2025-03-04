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
        self.cool_down_grid = _signal_info.cool_down_grid
        self.cool_down_ts = _signal_info.cool_down_ts
        self.signal_cache = {}

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

    def update_signals(self, _signal_ts, _signal_price, _data) -> list[Signal]:
        self.signal_mgr.update_state(_data)
        signals = self.signal_mgr.generate_signals(_data) or []
        return self.filter_signals(_signal_ts, _signal_price, signals)

    def filter_signals(self, _signal_ts, _signal_price, _signals: list[Signal]):
        filtered_signals = []
        for signal in _signals[::-1]:
            key = (signal.symbol, signal.direction)
            current_price = _signal_price[signal.symbol]
            current_ts = _signal_ts

            last_info = self.signal_cache.get(key) or [current_price, current_ts]
            last_price, last_ts = last_info

            delta_pct = abs(current_price - last_price) / (last_price + current_price) * 2
            condition_1 = delta_pct > self.cool_down_grid
            condition_2 = current_ts - last_ts > self.cool_down_ts
            if condition_1 and condition_2:
                signal.timestamp = _signal_ts
                signal.price_dict = _signal_price
                filtered_signals.append(signal)

            self.signal_cache[key] = [current_price, current_ts]

        return filtered_signals