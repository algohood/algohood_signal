# flake8: noqa
"""
@Create: 2024/9/30 14:06
@File: performanceMgr.py
@Author: Jijingyuan
"""
import importlib

from algoUtils.reloadUtil import reload_all

from algoUtils.schemaUtil import PerformanceMgrParam, Performance, EntryInfo, ExitInfo
from typing import Optional, Dict, List


class PerformanceMgr:

    def __init__(self, _performance_info: Optional[PerformanceMgrParam]):
        self.performance_info = _performance_info
        self.performance_mgr = self.get_performance_method(_performance_info)
        self.entry_info: Optional[EntryInfo] = None
        self.exit_info: Optional[ExitInfo] = None

    @staticmethod
    def get_performance_method(_method_info: Optional[PerformanceMgrParam]):
        if _method_info is None:
            return
        
        module = importlib.import_module('algoStrategy.algoPerformances.{}'.format(_method_info.performance_method_name))
        reload_all(module)
        cls_method = getattr(module, 'Algo')
        if cls_method is None:
            raise Exception('Unknown Method: {}'.format(_method_info.performance_method_name))
        
        instance = cls_method(**_method_info.performance_method_param)
        return instance

    def generate_performance(self) -> Optional[Performance]:
        if self.entry_info.entry_direction == 'long':
            is_win = self.exit_info.exit_price > self.entry_info.entry_price
            trade_ret = self.exit_info.exit_price / self.entry_info.entry_price - 1
            trade_ret -= self.performance_info.entry_fee + self.performance_info.exit_fee
        else:
            is_win = self.exit_info.exit_price < self.entry_info.entry_price
            trade_ret = self.entry_info.entry_price / self.exit_info.exit_price - 1
            trade_ret -= self.performance_info.entry_fee + self.performance_info.exit_fee

        return Performance(
            success=True,
            is_win=is_win,
            trade_ret=trade_ret,
            trade_duration=self.exit_info.exit_timestamp - self.entry_info.entry_timestamp
        )

    def update_performance(self, _data: Dict[str, List[List]]) -> Optional[Performance]:
        if self.performance_mgr is None:
            return
        
        if self.entry_info is None:
            self.entry_info = self.performance_mgr.generate_entry_info(_data)
        
        if self.entry_info is not None:
            if self.entry_info.entry_success is True:
                self.exit_info = self.performance_mgr.generate_exit_info(_data)
                if self.exit_info is not None:
                    return self.generate_performance()

            else:
                return Performance(success=False)
