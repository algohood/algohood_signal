# flake8: noqa
"""
@Create: 2024/9/30 14:06
@File: eventMgr.py
@Author: Jijingyuan
"""
import asyncio
import traceback
import time
from itertools import groupby
from typing import Dict, List
from ..algoConfig.loggerConfig import logger
from ..algoEngine.signalMgr import SignalMgr
from ..algoEngine.featureMgr import FeatureMgr
from .modelMgr import ModelMgr
from ..algoEngine.targetMgr import TargetMgr
from algoUtils.baseUtil import Sample


class EventMgr:
    def __init__(
            self, _signal_mgr, _feature_mgr, _target_mgr, _model_mgr, _data_mgr
    ):
        self.signal_mgr: SignalMgr = _signal_mgr
        self.feature_mgr: FeatureMgr = _feature_mgr
        self.target_mgr: TargetMgr = _target_mgr
        self.model_mgr: ModelMgr = _model_mgr
        self.data_mgr = _data_mgr
        self.cache = []
        self.cache_samples = []
        self.profile_records = {
            'target_mgr.get_completed_samples': 0,
            'model_mgr.update_model': 0,
            'feature_mgr.update_features': 0,
            'signal_mgr.update_signals': 0,
            'model_mgr.predict_target': 0,
            'target_mgr.add_sample': 0,
            'total_cost': 0,
        }

    @staticmethod
    def format_samples(_samples: list[Sample]):
        result = []
        for sample in _samples:
            result.append({
                **sample.signal.model_dump(exclude_defaults=True), 
                **(sample.features or {}), 
                **(sample.targets or {}), 
                **({'forecast': sample.sim_intercept} if sample.sim_intercept is not None else {}),
                **({'actual': sample.intercept} if sample.intercept is not None else {}),
                **(sample.performance.model_dump(exclude_defaults=True) if sample.performance is not None else {}),
            })

        return result

    async def start_task(self, _lag, _symbols, _start_timestamp, _end_timestamp):
        tasks = [
            self.data_mgr.load_data(_symbols, _start_timestamp, _end_timestamp),
            self.handle_data(_lag)
        ]

        await asyncio.gather(*tasks)
        logger.info(self.profile_records)
        return self.format_samples(self.cache_samples)

    @staticmethod
    def reshape(_ticks, _lag):
        keep = len(str(_lag).split('.')[-1]) if _lag < 1 else 0
        g = groupby(_ticks, lambda x: round(int(x[0] / _lag) * _lag, keep))
        return {k: list(v) for k, v in g}

    def handle_batch_data(self, _signal_ts, _signal_price, _data: Dict[str, List[List]]):
        t0 = time.time()
        completed_samples = self.target_mgr.get_completed_samples(_data)
        self.profile_records['target_mgr.get_completed_samples'] += time.time() - t0

        if completed_samples:
            t0 = time.time()
            self.model_mgr.update_model(completed_samples)
            self.profile_records['model_mgr.update_model'] += time.time() - t0
        
        t0 = time.time()
        self.feature_mgr.update_features(_data)
        self.profile_records['feature_mgr.update_features'] += time.time() - t0

        t0 = time.time()
        signals = self.signal_mgr.update_signals(_signal_ts, _signal_price, _data)
        self.profile_records['signal_mgr.update_signals'] += time.time() - t0    

        if signals:
            t0 = time.time()
            features = self.feature_mgr.generate_features(_data) or {}
            sim_targets = self.model_mgr.predict_target(features)
            self.profile_records['model_mgr.predict_target'] += time.time() - t0

            t0 = time.time()
            for signal in signals:
                sample = Sample(
                    signal=signal, 
                    features=features, 
                    sim_targets=sim_targets
                )
                self.target_mgr.add_sample(sample)
                self.cache_samples.append(sample)

            self.profile_records['target_mgr.add_sample'] += time.time() - t0

    async def handle_data(self, _lag):
        while True:
            try:
                data = await self.data_mgr.get_data()
                if data is None:
                    return

                t0 = time.time()
                current_data = self.cache + data
                if _lag is None:
                    for signal_data in current_data:
                        signal_ts = signal_data[0]
                        signal_price = {signal_data[1]: signal_data[2][2]}
                        self.handle_batch_data(signal_ts, signal_price, {signal_data[1]: [signal_data[2]]})

                else:
                    last_cut = None
                    last_ticks = []
                    for cut_timestamp, ticks in self.reshape(current_data, _lag).items():
                        if last_cut is None:
                            last_cut = cut_timestamp
                            last_ticks = ticks 
                            continue

                        signal_data = {}
                        for v in last_ticks:
                            signal_data.setdefault(v[1], []).append(v[2])

                        signal_ts = round(last_cut + _lag, 6)
                        signal_price = {k: v[-1][2] for k, v in signal_data.items()}
                        self.handle_batch_data(signal_ts, signal_price, signal_data)

                        last_cut = cut_timestamp
                        last_ticks = ticks

                    self.cache = last_ticks

                t1 = time.time()
                self.profile_records['total_cost'] += t1 - t0

            except Exception:
                logger.error(traceback.format_exc())

    @staticmethod
    def check_fields(_signal):
        default_keys = ['batch_id', 'symbol', 'action', 'position']
        signal_keys = list(_signal.keys())
        for key in default_keys:
            if key not in signal_keys:
                raise Exception('{} does not exist'.format(key))
