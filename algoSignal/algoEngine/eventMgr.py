# flake8: noqa
"""
@Create: 2024/9/30 14:06
@File: eventMgr.py
@Author: Jijingyuan
"""
import asyncio
import traceback
from itertools import groupby

from ..algoConfig.loggerConfig import logger
from ..algoEngine.signalMgr import SignalMgr
from ..algoEngine.featureMgr import FeatureMgr
from ..algoEngine.interceptMgr import InterceptMgr
from ..algoEngine.targetMgr import TargetMgr
from algoUtils.baseUtil import Sample


class EventMgr:
    def __init__(
            self, _signal_mgr, _feature_mgr, _target_mgr, _intercept_mgr, _data_mgr
    ):
        self.signal_mgr: SignalMgr = _signal_mgr
        self.feature_mgr: FeatureMgr = _feature_mgr
        self.target_mgr: TargetMgr = _target_mgr
        self.intercept_mgr: InterceptMgr = _intercept_mgr
        self.data_mgr = _data_mgr
        self.cache = []
        self.features = {}
        self.cache_samples = []

    @staticmethod
    def format_samples(_samples: list[Sample]):
        result = []
        for sample in _samples:
            result.append({
                **sample.signal.model_dump(exclude_defaults=True), 
                **sample.features, 
                **sample.targets, 
                'forecast': sample.forecast, 
                'actual': sample.actual
            })

        return result


    async def start_task(self, _lag, _symbols, _start_timestamp, _end_timestamp):
        tasks = [
            self.data_mgr.load_data(_symbols, _start_timestamp, _end_timestamp),
            self.handle_data(_lag)
        ]

        await asyncio.gather(*tasks)
        return self.format_samples(self.cache_samples)

    @staticmethod
    def reshape(_ticks, _lag):
        keep = len(str(_lag).split('.')[-1]) if _lag < 1 else 0
        g = groupby(_ticks, lambda x: round(int(x[0] / _lag) * _lag, keep))
        return {k: list(v) for k, v in g}

    def handle_batch_data(self, _signal_ts, _signal_price, _data: dict):
        self.features.update(self.feature_mgr.update_features(_data))

        signals = self.signal_mgr.update_signals(_data)
        if signals:
            is_intercepted = self.intercept_mgr.handle_forecast(self.features)
            for signal in signals:
                signal.timestamp = _signal_ts
                signal.price_dict = _signal_price
                sample = Sample(signal=signal, features=self.features, forecast=is_intercepted)
                self.target_mgr.add_sample(sample)
                self.cache_samples.append(sample)

        completed_samples = self.target_mgr.get_completed_samples(_data)
        if completed_samples:
            self.intercept_mgr.update_model(completed_samples)
            for sample in completed_samples:
                sample.actual = self.intercept_mgr.handle_actual(sample.targets)

    async def handle_data(self, _lag):
        while True:
            try:
                data = await self.data_mgr.get_data()
                if data is None:
                    return

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

            except Exception:
                logger.error(traceback.format_exc())

    @staticmethod
    def check_fields(_signal):
        default_keys = ['batch_id', 'symbol', 'action', 'position']
        signal_keys = list(_signal.keys())
        for key in default_keys:
            if key not in signal_keys:
                raise Exception('{} does not exist'.format(key))
