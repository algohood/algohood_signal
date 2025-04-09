# flake8: noqa
"""
@Create: 2024/9/30 14:06
@File: performanceMgr.py
@Author: Jijingyuan
"""
import importlib
import time
import numpy as np
from algoUtils.reloadUtil import reload_all
from algoUtils.schemaUtil import *
from algoUtils.baseUtil import PerformanceBase
from typing import Optional, Dict, List
from algoSignal.algoConfig.profileConfig import profile_stats
from algoSignal.algoConfig.loggerConfig import logger


class PerformanceMgr:

    def __init__(self, _performance_info: PerformanceMgrParam, _data_mgr):
        self.performance_info = _performance_info
        self.data_mgr = _data_mgr
        self.symbols = []
        self.start_timestamp = 0
        self.end_timestamp = 0
        self.performance_q = []
        self.trade_result_q = []

    def get_performance_method(self, _signal: Signal) -> PerformanceBase:
        module = importlib.import_module('algoStrategy.algoPerformances.{}'.format(self.performance_info.performance_method_name))
        reload_all(module)
        cls_method = getattr(module, 'Algo')
        if cls_method is None:
            raise Exception('Unknown Method: {}'.format(self.performance_info.performance_method_name))
        
        instance = cls_method(**self.performance_info.performance_method_param)
        instance.init_signal(_signal)
        return instance
    
    def init_signals(self, _symbols: List[str], _start_timestamp: float, _end_timestamp: float, _signals: List[Signal]):
        logger.info('start init signals')
        self.symbols = _symbols
        self.start_timestamp = _start_timestamp
        self.end_timestamp = _end_timestamp
        _signals.sort(key=lambda x: x.timestamp, reverse=True)
        self.performance_q = [(signal, self.get_performance_method(signal)) for signal in _signals]

    def generate_performance(self, _data: Dict[str, np.ndarray]) -> Optional[Union[float, str]]:
        last_data_ts = max([v[-1, 0] for v in _data.values()])
        total_len = len(self.performance_q)
        for i in range(total_len - 1, -1, -1):
            signal, performance = self.performance_q[i]
            if signal.timestamp > last_data_ts:
                break

            trading_result = performance.generate_trading_result(_data)
            if trading_result is None:
                continue
            
            self.trade_result_q.append((signal, trading_result))
            self.performance_q.pop(i)

        if not self.performance_q:
            return 'performance over'

        last_signal = self.performance_q[-1][0]
        if last_signal.timestamp < last_data_ts:
            return

        return last_signal.timestamp

    async def start_task(self, _generate_abstract=True):
        logger.info('start generate performances')
        t0 = time.time()
        profile_stats.reset()
        await self.data_mgr.load_data_call_back(self.generate_performance, self.symbols, self.start_timestamp, self.end_timestamp)
        logger.info(f"task time: {time.time() - t0}")
        profile_stats.show_stats()

        performance_list = []
        for signal, trading_result in self.trade_result_q:
            signal_dict = signal.model_dump(exclude_defaults=True)
            other_info = signal_dict.pop('other_info', {})
            trading_result_dict = trading_result.model_dump(exclude_defaults=True)

            performance_list.append({
                **signal_dict,
                **trading_result_dict,
                **other_info,
            })

        performance_abstract = self.generate_performance_abstract(performance_list) if _generate_abstract else {}
        return performance_list, performance_abstract

    @staticmethod
    def generate_performance_abstract(_performance_list):
        if not _performance_list:
            return {}

        # 获取成功的交易
        successful_trades = [trade for trade in _performance_list if trade.get('success', False)]
        
        # 按时间排序的交易列表
        time_sorted_trades = sorted(successful_trades, key=lambda x: x.get('entry_timestamp', 0))
        
        if not time_sorted_trades:
            return {}
            
        # 计算交易时间范围
        start_time = time_sorted_trades[0].get('entry_timestamp', 0)
        end_time = time_sorted_trades[-1].get('exit_timestamp', 0) if time_sorted_trades[-1].get('exit_timestamp') else time_sorted_trades[-1].get('entry_timestamp', 0)
        trading_period = end_time - start_time if end_time > start_time else 1
        
        #
        # 第一部分：整体表现统计
        #
        
        # 基本统计
        total_trades = len(_performance_list)
        successful_trades_count = len(successful_trades)
        success_rate = successful_trades_count / total_trades if total_trades > 0 else 0
        
        # 盈利统计
        win_trades = sum(1 for trade in successful_trades if trade.get('is_win', 0) > 0)
        win_rate = win_trades / successful_trades_count if successful_trades_count > 0 else 0
        
        # 收益统计
        returns = [trade.get('trade_ret', 0) for trade in successful_trades]
        total_return = sum(returns) if returns else 0
        
        # 计算平均盈亏比
        profit_returns = [trade.get('trade_ret', 0) for trade in successful_trades if trade.get('is_win', 0) > 0]
        loss_returns = [trade.get('trade_ret', 0) for trade in successful_trades if trade.get('is_win', 0) == 0]
        
        avg_profit = sum(profit_returns) / len(profit_returns) if profit_returns else 0
        avg_loss = sum(loss_returns) / len(loss_returns) if loss_returns else 0
        profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0
        
        # 计算最大回撤
        cumulative_returns = []
        cum_return = 0
        
        for trade in time_sorted_trades:
            ret = trade.get('trade_ret', 0)
            cum_return += ret
            cumulative_returns.append(cum_return)
        
        max_drawdown = 0
        peak = 0
        for cum_ret in cumulative_returns:
            if cum_ret > peak:
                peak = cum_ret
            drawdown = peak - cum_ret
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # 计算平均持续时间
        durations = [trade.get('trade_duration', 0) for trade in successful_trades]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        #
        # 第二部分：历史最佳收益时段的统计（最大连续上涨区间）
        #
        
        # 寻找历史中最大的连续上涨区间（优化算法，减少循环）
        max_gain = 0
        start_idx = 0
        end_idx = 0
        has_uptrend = False
        
        if len(cumulative_returns) > 1:
            # 使用动态规划思想优化查找
            current_start = 0
            for i in range(1, len(cumulative_returns)):
                # 如果当前值比前一个高，可能是上涨中
                if cumulative_returns[i] > cumulative_returns[i-1]:
                    has_uptrend = True
                    gain = cumulative_returns[i] - cumulative_returns[current_start]
                    # 如果找到更大的收益区间，更新最大上涨区间
                    if gain > max_gain:
                        max_gain = gain
                        start_idx = current_start
                        end_idx = i
                # 如果当前值低于前一个，记录为潜在的新起点
                elif cumulative_returns[i] < cumulative_returns[i-1]:
                    # 如果当前点比起点还低，更新起点
                    if cumulative_returns[i] < cumulative_returns[current_start]:
                        current_start = i
        
        # 如果没有找到明显的上涨，将所有相关指标置为零
        if not has_uptrend:
            historical_max_return = 0
            max_return_trades = []
            max_return_trades_count = 0
            max_return_duration = 0
            max_return_win_rate = 0
            max_return_profit_loss_ratio = 0
            max_return_max_drawdown = 0
        else:
            historical_max_return = max_gain
            
            # 历史最大收益时段的统计
            max_return_trades = time_sorted_trades[start_idx:end_idx+1] if end_idx >= start_idx else []
            max_return_trades_count = len(max_return_trades)
            
            # 计算持续时间
            max_return_duration = 0
            if max_return_trades:
                best_start_time = max_return_trades[0].get('entry_timestamp', 0)
                best_end_time = max_return_trades[-1].get('exit_timestamp', 0) if max_return_trades[-1].get('exit_timestamp') else max_return_trades[-1].get('entry_timestamp', 0)
                max_return_duration = best_end_time - best_start_time if best_end_time > best_start_time else 0
            
            # 计算胜率
            max_return_wins = sum(1 for trade in max_return_trades if trade.get('is_win', 0) > 0)
            max_return_win_rate = max_return_wins / max_return_trades_count if max_return_trades_count > 0 else 0
            
            # 计算盈亏比
            max_return_profit = [trade.get('trade_ret', 0) for trade in max_return_trades if trade.get('is_win', 0) > 0]
            max_return_loss = [trade.get('trade_ret', 0) for trade in max_return_trades if trade.get('is_win', 0) == 0]
            
            max_return_avg_profit = sum(max_return_profit) / len(max_return_profit) if max_return_profit else 0
            max_return_avg_loss = sum(max_return_loss) / len(max_return_loss) if max_return_loss else 0
            max_return_profit_loss_ratio = abs(max_return_avg_profit / max_return_avg_loss) if max_return_avg_loss != 0 else 0
            
            # 简化回撤计算逻辑
            max_return_max_drawdown = 0
            max_return_peak = cumulative_returns[start_idx] if start_idx < len(cumulative_returns) else 0
            
            # 单次遍历计算回撤
            for i in range(start_idx, end_idx + 1):
                if i < len(cumulative_returns):
                    if cumulative_returns[i] > max_return_peak:
                        max_return_peak = cumulative_returns[i]
                    drawdown = max_return_peak - cumulative_returns[i]
                    max_return_max_drawdown = max(max_return_max_drawdown, drawdown)
            
            # 确保最大回撤不超过区间总收益
            max_return_max_drawdown = min(max_return_max_drawdown, historical_max_return)
        
        #
        # 第三部分：前20%最好的交易时间段表现
        #
        
        # 将交易时间分成小时级别的时间段
        hour_sec = 3600  # 一小时的秒数
        total_hours = int((end_time - start_time) / hour_sec) + 1
        
        # 创建每小时的时间段
        time_segments = []
        for i in range(total_hours):
            segment_start = start_time + i * hour_sec
            segment_end = segment_start + hour_sec
            
            # 该时间段内的交易
            segment_trades = [
                trade for trade in time_sorted_trades 
                if segment_start <= trade.get('entry_timestamp', 0) < segment_end
            ]
            
            # 计算该时间段的收益
            segment_return = sum(trade.get('trade_ret', 0) for trade in segment_trades)
            
            time_segments.append({
                'start_time': segment_start,
                'end_time': segment_end,
                'trades': segment_trades,
                'return': segment_return
            })
        
        # 按收益排序时间段，只保留有交易的时间段
        active_segments = [segment for segment in time_segments if segment['trades']]
        active_segments.sort(key=lambda x: x['return'], reverse=True)
        
        # 前20%收益最好的时间段
        top20_count = max(1, int(len(active_segments) * 0.2))
        top20_segments = active_segments[:top20_count]
        rest80_segments = active_segments[top20_count:]
        
        # 提取前20%时间段内的所有交易
        top20_trades = []
        for segment in top20_segments:
            top20_trades.extend(segment['trades'])
            
        # 按时间顺序排序
        top20_trades = sorted(top20_trades, key=lambda x: x.get('entry_timestamp', 0))
        top20_count = len(top20_trades)
        
        # 计算总收益
        top20_return = sum(trade.get('trade_ret', 0) for trade in top20_trades)
        
        # 计算胜率
        top20_wins = sum(1 for trade in top20_trades if trade.get('is_win', 0) > 0)
        top20_win_rate = top20_wins / top20_count if top20_count > 0 else 0
        
        # 计算盈亏比
        top20_profit = [trade.get('trade_ret', 0) for trade in top20_trades if trade.get('is_win', 0) > 0]
        top20_loss = [trade.get('trade_ret', 0) for trade in top20_trades if trade.get('is_win', 0) == 0]
        
        top20_avg_profit = sum(top20_profit) / len(top20_profit) if top20_profit else 0
        top20_avg_loss = sum(top20_loss) / len(top20_loss) if top20_loss else 0
        top20_profit_loss_ratio = abs(top20_avg_profit / top20_avg_loss) if top20_avg_loss != 0 else 0
        
        # 计算最大回撤（按时间顺序计算）
        top20_cum_returns = []
        top20_cum_return = 0
        
        for trade in top20_trades:
            ret = trade.get('trade_ret', 0)
            top20_cum_return += ret
            top20_cum_returns.append(top20_cum_return)
            
        top20_max_drawdown = 0
        top20_peak = 0
        
        for cum_ret in top20_cum_returns:
            if cum_ret > top20_peak:
                top20_peak = cum_ret
            drawdown = top20_peak - cum_ret
            if drawdown > top20_max_drawdown:
                top20_max_drawdown = drawdown
                
        #
        # 第四部分：剩余80%交易时间的表现
        #
        
        # 提取剩余80%时间段内的所有交易
        rest80_trades = []
        for segment in rest80_segments:
            rest80_trades.extend(segment['trades'])
            
        # 按时间顺序排序
        rest80_trades = sorted(rest80_trades, key=lambda x: x.get('entry_timestamp', 0))
        rest80_count = len(rest80_trades)
        
        # 计算总收益
        rest80_return = sum(trade.get('trade_ret', 0) for trade in rest80_trades)
            
        # 计算胜率
        rest80_wins = sum(1 for trade in rest80_trades if trade.get('is_win', 0) > 0)
        rest80_win_rate = rest80_wins / rest80_count if rest80_count > 0 else 0
        
        # 计算盈亏比
        rest80_profit = [trade.get('trade_ret', 0) for trade in rest80_trades if trade.get('is_win', 0) > 0]
        rest80_loss = [trade.get('trade_ret', 0) for trade in rest80_trades if trade.get('is_win', 0) == 0]
        
        rest80_avg_profit = sum(rest80_profit) / len(rest80_profit) if rest80_profit else 0
        rest80_avg_loss = sum(rest80_loss) / len(rest80_loss) if rest80_loss else 0
        rest80_profit_loss_ratio = abs(rest80_avg_profit / rest80_avg_loss) if rest80_avg_loss != 0 else 0
        
        # 计算最大回撤
        rest80_cum_returns = []
        rest80_cum_return = 0
        
        for trade in rest80_trades:
            ret = trade.get('trade_ret', 0)
            rest80_cum_return += ret
            rest80_cum_returns.append(rest80_cum_return)
            
        rest80_max_drawdown = 0
        rest80_peak = 0
        
        for cum_ret in rest80_cum_returns:
            if cum_ret > rest80_peak:
                rest80_peak = cum_ret
            drawdown = rest80_peak - cum_ret
            if drawdown > rest80_max_drawdown:
                rest80_max_drawdown = drawdown
        
        # 组装最终摘要
        abstract = {
            # 第一部分：整体表现统计
            'total_trades': total_trades,
            'successful_trades': successful_trades_count,
            'success_rate': success_rate,
            'win_rate': win_rate,
            'total_return': total_return,
            'profit_loss_ratio': profit_loss_ratio,
            'max_drawdown': max_drawdown,
            'avg_duration': avg_duration,
            'trading_period': trading_period,
            
            # 第二部分：历史最佳收益时段的统计（最大连续上涨区间）
            'best_return': historical_max_return,
            'best_trades_count': max_return_trades_count,
            'best_duration': max_return_duration / 60,
            'best_win_rate': max_return_win_rate,
            'best_profit_loss_ratio': max_return_profit_loss_ratio,
            'best_max_drawdown': max_return_max_drawdown,
            'best_ret_per_mdd': historical_max_return / max_return_max_drawdown,
            
            # 第三部分：前20%最好的交易时间段表现
            'top20_return': top20_return,
            'top20_trades_count': top20_count,
            'top20_win_rate': top20_win_rate,
            'top20_profit_loss_ratio': top20_profit_loss_ratio,
            'top20_max_drawdown': top20_max_drawdown,
            
            # 第四部分：剩余80%交易时间的表现
            'rest80_return': rest80_return,
            'rest80_trades_count': rest80_count,
            'rest80_win_rate': rest80_win_rate,
            'rest80_profit_loss_ratio': rest80_profit_loss_ratio,
            'rest80_max_drawdown': rest80_max_drawdown
        }
        
        return abstract


if __name__ == '__main__':
    import pandas as pd
    from pprint import pprint
    performance_list = pd.read_csv('../algoFile/1742715738425558_MakerFixTPFixSL/MakerFixTPFixSL/grid_doge_usdt_binance_future.csv').to_dict('records')
    performance_abstract = PerformanceMgr.generate_performance_abstract(performance_list)
    pprint(performance_abstract, sort_dicts=False)
