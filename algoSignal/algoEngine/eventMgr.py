# flake8: noqa
"""
@Create: 2024/9/30 14:06
@File: eventMgr.py
@Author: Jijingyuan
"""
import asyncio
import traceback
import time
import heapq
import numpy as np
from itertools import groupby
from typing import Dict, List
from ..algoConfig.loggerConfig import logger
from ..algoEngine.signalMgr import SignalMgr
from ..algoEngine.featureMgr import FeatureMgr
from .modelMgr import ModelMgr
from ..algoEngine.targetMgr import TargetMgr
from algoUtils.baseUtil import Sample
from ..algoConfig.profileConfig import profile_stats


class EventMgr:
    def __init__(
            self, _signal_mgr, _feature_mgr, _target_mgr, _model_mgr, _data_mgr
    ):
        self.signal_mgr: SignalMgr = _signal_mgr
        self.feature_mgr: FeatureMgr = _feature_mgr
        self.target_mgr: TargetMgr = _target_mgr
        self.model_mgr: ModelMgr = _model_mgr
        self.data_mgr = _data_mgr
        self.cache = {}
        self.cache_samples = []

    @staticmethod
    def format_samples(_samples: list[Sample]):
        result = []
        for sample in _samples:
            result.append({
                **sample.signal.model_dump(exclude_defaults=True), 
                **(sample.features or {}), 
                **(sample.targets or {}), 
                **({'forecast_intercept': sample.sim_intercept} if sample.sim_intercept is not None else {}),
                **({'actual_intercept': sample.intercept} if sample.intercept is not None else {}),
            })

        return result

    async def start_task(self, _lag, _symbols, _start_timestamp, _end_timestamp):
        t0 = time.time()
        profile_stats.reset()
        tasks = [
            self.data_mgr.load_data(_symbols, _start_timestamp, _end_timestamp),
            self.handle_data(_lag)
        ]

        await asyncio.gather(*tasks)
        logger.info(f"task time: {time.time() - t0}")
        profile_stats.show_stats()
        return self.format_samples(self.cache_samples)

    @staticmethod
    @profile_stats.update_cost_time
    def reshape(_ticks_dict: Dict[str, np.ndarray], _lag) -> Dict[float, Dict[str, np.ndarray]]:
        # 预先创建结果字典
        result = {}
        
        # 直接计算保留小数位数
        keep = 6  # 因为时间戳保留六位小数
        round_factor = 10 ** keep
        
        # 处理每个symbol的数据
        for symbol, ticks in _ticks_dict.items():
            if len(ticks) == 0:
                continue
            
            # 提取时间戳列
            timestamps = ticks[:, 0]
            
            # 向量化计算分组键 - 这里使用更高效的向量化操作
            group_keys = np.ceil(timestamps / _lag) * _lag
            # 确保保留六位小数
            group_keys = np.round(group_keys * round_factor) / round_factor
            
            # 找出唯一键的变化位置
            changes = np.where(np.diff(np.append(group_keys, [np.inf])))[0]
            
            # 一次性处理所有切片
            start_idx = 0
            for end_idx in changes:
                key = float(group_keys[start_idx])
                # 使用字典的setdefault避免重复检查
                batch = result.setdefault(key, {})
                batch[symbol] = ticks[start_idx:end_idx+1]
                start_idx = end_idx + 1
        
        # 返回按键排序的结果
        return {k: result[k] for k in sorted(result)}

    @profile_stats.update_cost_time
    def handle_batch_data(self, _current_ts: float, _data: Dict[str, np.ndarray]):
        completed_samples = self.target_mgr.get_completed_samples(_current_ts, _data)
        if completed_samples:
            self.model_mgr.update_model(completed_samples)

        self.feature_mgr.update_features(_current_ts, _data)
        signal = self.signal_mgr.update_signals(_current_ts, _data)

        if signal:
            features = self.feature_mgr.generate_features(_current_ts, _data) or {}
            sim_targets = self.model_mgr.predict_target(features)
            sample = Sample(
                signal=signal, 
                features=features, 
                sim_targets=sim_targets
            )
            self.target_mgr.add_sample(sample)
            self.cache_samples.append(sample)

    async def handle_data(self, _lag):
        while True:
            try:
                data = await self.data_mgr.get_data()
                if data is None:
                    return

                if _lag is None:
                    heap = []
                    iter_dict = {k: iter(v) for k, v in data.items()}
                    for symbol, it in iter_dict.items():
                        row = next(it)
                        heapq.heappush(heap, (row[0], symbol, row))

                    while heap:
                        current_ts, symbol, row = heapq.heappop(heap)
                        row_2d = np.array([row])
                        self.handle_batch_data(current_ts, {symbol: row_2d})
                        try:
                            row = next(iter_dict[symbol])
                            heapq.heappush(heap, (row[0], symbol, row))
                        except StopIteration:
                            pass

                else:
                    symbols = list(data.keys())
                    for symbol in symbols:
                        batch = data[symbol]
                        cache = self.cache.get(symbol)
                        if cache is None:
                            continue
                        
                        data[symbol] = np.concatenate([cache, batch])

                    last_cut = None
                    last_ticks = {}
                    for cut_timestamp, ticks in self.reshape(data, _lag).items():
                        if last_cut is None:
                            last_cut = cut_timestamp
                            last_ticks = ticks 
                            continue

                        self.handle_batch_data(last_cut, last_ticks)
                        last_cut = cut_timestamp
                        last_ticks = ticks

                    self.cache = last_ticks

            except Exception:
                logger.error(traceback.format_exc())
