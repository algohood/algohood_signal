# flake8: noqa
"""
@Create: 2024/9/30 14:06
@File: dataMgr.py
@Author: Jijingyuan
"""
import numpy as np
import asyncio
from asyncio import Queue

from algoUtils.asyncRedisUtil import AsyncRedisClient
from algoUtils.dateUtil import timestamp_local_datetime
from .clusterMgr import RedisCluster
from ..algoConfig.loggerConfig import logger
from ..algoConfig.profileConfig import profile_stats
from typing import Optional


class DataMgr:
    NODE_LIMIT = 50000

    def __init__(self, _host, _port):
        self.config_redis = AsyncRedisClient(_host, _port)
        self.redis_cluster = RedisCluster(self.config_redis)

        self.cache_data = {}
        self.ticks_q = Queue()
        self.data_type = None

    async def init_data_mgr(self, _is_localhost):
        await self.redis_cluster.init_cluster(_is_localhost=_is_localhost)
        logger.info('data mgr initiated')

    def clear_cache(self):
        self.ticks_q = Queue()
        self.cache_data = {}

    def set_data_type(self, _data_type):
        self.data_type = _data_type

    @profile_stats.update_cost_time
    async def load_cache_data(self, _symbol, _start_timestamp):
        if _symbol in self.cache_data:
            return

        pair, exchange = _symbol.split('|')
        labels = {'pair': pair, 'exchange': exchange, 'data_type': self.data_type}
        rsp = await self.redis_cluster.get_batch_by_labels(0, _start_timestamp, '+', labels, self.NODE_LIMIT)
        if rsp is None:
            logger.error('cluster node abnormal')

        self.cache_data[_symbol] = self.format_node_data(rsp)

    @profile_stats.update_cost_time
    def merge_data(self, _symbol, _check_timestamp, _end_timestamp):
        array_data = self.cache_data.pop(_symbol)
        if _check_timestamp > _end_timestamp:
            _check_timestamp = _end_timestamp
            cache_dict = {_symbol: array_data[array_data[:, 0] <= _end_timestamp]}
        else:
            cache_dict = {_symbol: array_data}

        left_symbols = list(self.cache_data.keys())
        while left_symbols:
            symbol = left_symbols.pop(0)
            batch = self.cache_data[symbol]
            if batch is None:
                self.cache_data.pop(symbol)
                continue

            if batch.size < 1000:
                self.cache_data.pop(symbol)

            cache_dict[symbol] = batch[batch[:, 0] <= _check_timestamp]
            self.cache_data[symbol] = batch[batch[:, 0] > _check_timestamp]

        return {k: v for k, v in cache_dict.items() if v.size > 0}

    async def load_data_call_back(self, _callback, _symbols, _start_timestamp, _end_timestamp):
        logger.info('start receiving offline data')
        cut_timestamp = _start_timestamp
        symbols = [_symbols] if isinstance(_symbols, str) else _symbols
        tasks = []
        while symbols:
            tasks = [self.load_cache_data(symbol, cut_timestamp) for symbol in symbols]
            await asyncio.gather(*tasks)
            
            check_timestamp = float('inf')
            out_symbol = None
            for symbol, batch in self.cache_data.items():
                if batch is None:
                    symbols.remove(symbol)
                    continue

                if batch[-1][0] < check_timestamp:
                    check_timestamp = batch[-1][0]
                    out_symbol = symbol

            if out_symbol is None:
                logger.info('data out of range')
                return

            merged_data = self.merge_data(out_symbol, check_timestamp, _end_timestamp)
            if merged_data:
                logger.info('merged data sync from {} to {}: {}'.format(
                    timestamp_local_datetime(cut_timestamp),
                    timestamp_local_datetime(check_timestamp), 
                    sum([len(v) for v in merged_data.values()])
                ))
                tmp = _callback(merged_data)
                if isinstance(tmp, float):
                    self.clear_cache()
                    cut_timestamp = tmp
                    continue

                elif tmp == 'performance over':
                    logger.info('performance over')
                    return

            cut_timestamp = check_timestamp + 0.000001
            if check_timestamp >= _end_timestamp:
                logger.info('data over')
                return

        logger.info('data insufficient')

    async def load_data(self, _symbols, _start_timestamp, _end_timestamp):
        logger.info('start receiving offline data')
        cut_timestamp = _start_timestamp
        symbols = [_symbols] if isinstance(_symbols, str) else _symbols
        tasks = []
        while symbols:
            tasks = [self.load_cache_data(symbol, cut_timestamp) for symbol in symbols]
            await asyncio.gather(*tasks)
            
            check_timestamp = float('inf')
            out_symbol = None
            for symbol, batch in self.cache_data.items():
                if batch is None:
                    symbols.remove(symbol)
                    continue

                if batch[-1][0] < check_timestamp:
                    check_timestamp = batch[-1][0]
                    out_symbol = symbol

            if out_symbol is None:
                logger.info('data out of range')
                self.ticks_q.put_nowait(None)
                return

            merged_data = self.merge_data(out_symbol, check_timestamp, _end_timestamp)
            cut_timestamp = check_timestamp + 0.000001
            if merged_data:
                logger.info('merged data sync to {}: {}'.format(
                    timestamp_local_datetime(check_timestamp), sum([len(v) for v in merged_data.values()])
                ))
                self.ticks_q.put_nowait(merged_data)

            if check_timestamp >= _end_timestamp:
                logger.info('data over')
                self.ticks_q.put_nowait(None)
                return

        logger.info('data insufficient')
        self.ticks_q.put_nowait(None)

    async def get_data(self) -> Optional[dict[str, np.ndarray]]:
        return await self.ticks_q.get()

    @staticmethod
    def format_node_data(_cluster_rsp) -> Optional[np.ndarray]:
        all_data = []
        last_ts = float('inf')
        for tmp in _cluster_rsp.values():
            values = [list(v.values())[0][1] for v in tmp]
            for info in zip(*values):
                recv_ts = round(info[0][0] / 1000000, 6)
                exchange_ts = round(info[3][1] / 1000000, 6)
                all_data.append([recv_ts, exchange_ts, info[1][1], info[0][1], int(info[2][1])])

            if all_data:
                last_ts = min(last_ts, all_data[-1][0])

        if not all_data:
            return

        array_data = np.array(all_data)
        data = array_data[array_data[:, 0] <= last_ts]
        return data[np.argsort(data[:, 0])]
