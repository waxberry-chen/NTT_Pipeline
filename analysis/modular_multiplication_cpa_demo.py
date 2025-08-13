#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模乘模块CPA攻击演示脚本

本脚本演示了如何对NTT流水线中的模乘模块进行CPA攻击，
包括功耗曲线读取、功耗模型构建和相关性分析。

作者: CPA攻击分析
日期: 2024
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import random
from typing import List, Tuple, Dict

class ModularMultiplicationCPA:
    """
    模乘模块CPA攻击类
    
    实现对NTT流水线中模乘运算的侧信道攻击
    """
    
    def __init__(self, 
                 power_traces_file: str,
                 target_key: int = 1000,
                 sample_points: int = 5000,
                 max_traces: int = 10000):
        """
        初始化CPA攻击参数
        
        Args:
            power_traces_file: 功耗迹线文件路径
            target_key: 目标密钥值
            sample_points: 每条迹线的采样点数
            max_traces: 最大使用的迹线数量
        """
        self.power_traces_file = power_traces_file
        self.target_key = target_key
        self.sample_points = sample_points
        self.max_traces = max_traces
        
        # 数据存储
        self.plaintexts = []
        self.power_traces = []
        self.loaded_traces = 0
        
        print(f"初始化模乘模块CPA攻击")
        print(f"目标密钥: {target_key} (二进制: {bin(target_key)})")
        print(f"采样点数: {sample_points}")
        print(f"最大迹线数: {max_traces}")
    
    def load_power_traces(self) -> bool:
        """
        加载功耗迹线数据
        
        Returns:
            bool: 加载是否成功
        """
        print("\n=== 加载功耗迹线数据 ===")
        
        if not os.path.exists(self.power_traces_file):
            print(f"错误: 功耗迹线文件不存在: {self.power_traces_file}")
            return False
        
        try:
            with open(self.power_traces_file, 'r') as f:
                for line_num, line in enumerate(f):
                    if self.loaded_traces >= self.max_traces:
                        break
                    
                    if not line.strip():
                        continue
                    
                    try:
                        # 解析数据格式: "明文:功耗迹线"
                        plaintext_str, power_trace_str = line.split(':', 1)
                        plaintext = int(plaintext_str.strip(), 10)  # 十进制明文
                        
                        # 解析功耗迹线
                        power_values = power_trace_str.strip().split()
                        power_trace = np.array(power_values, dtype=np.float64)
                        
                        # 调整迹线长度
                        if len(power_trace) < self.sample_points:
                            power_trace = np.pad(power_trace, 
                                               (0, self.sample_points - len(power_trace)))
                        elif len(power_trace) > self.sample_points:
                            power_trace = power_trace[:self.sample_points]
                        
                        self.plaintexts.append(plaintext)
                        self.power_traces.append(power_trace)
                        self.loaded_traces += 1
                        
                        if self.loaded_traces % 1000 == 0:
                            print(f"已加载 {self.loaded_traces} 条迹线")
                    
                    except Exception as e:
                        print(f"解析第 {line_num+1} 行时出错: {str(e)}")
                        continue
            
            print(f"成功加载 {self.loaded_traces} 条功耗迹线")
            return True
            
        except Exception as e:
            print(f"加载功耗迹线时出错: {str(e)}")
            return False
    
    def hamming_weight_model(self, plaintext: int, key_guess: int) -> int:
        """
        汉明重量功耗模型
        
        Args:
            plaintext: 明文值
            key_guess: 密钥猜测值
            
        Returns:
            int: 理论功耗值(汉明重量)
        """
        # 模乘运算的中间值
        intermediate = (plaintext * key_guess) % 8380417
        return bin(intermediate).count('1')
    
    def bit_flip_model(self, plaintext: int, key_guess: int) -> int:
        """
        位翻转功耗模型
        
        Args:
            plaintext: 明文值
            key_guess: 密钥猜测值
            
        Returns:
            int: 位翻转数量
        """
        # 计算运算前后的位翻转
        before = plaintext
        after = (plaintext * key_guess) % 8380417
        return bin(before ^ after).count('1')
    
    def ntt_specific_model(self, plaintext: int, key_guess: int) -> float:
        """
        NTT特定功耗模型
        
        Args:
            plaintext: 明文值
            key_guess: 密钥猜测值
            
        Returns:
            float: NTT特定功耗值
        """
        # NTT运算的特定模式
        result = (plaintext * key_guess) % 8380417
        
        # 考虑模约简的影响
        reduction_factor = 1.0 if result < plaintext * key_guess else 1.5
        
        # 考虑位模式
        bit_pattern = bin(result).count('01') + bin(result).count('10')
        
        return bin(result).count('1') * reduction_factor + bit_pattern * 0.1
    
    def perform_cpa_attack(self, 
                          power_models: Dict[str, callable],
                          key_range: Tuple[int, int] = (0, 8)) -> Dict[str, Dict]:
        """
        执行CPA攻击
        
        Args:
            power_models: 功耗模型字典
            key_range: 密钥搜索范围
            
        Returns:
            Dict: 攻击结果
        """
        print("\n=== 执行CPA攻击 ===")
        
        if not self.power_traces:
            print("错误: 没有加载功耗迹线数据")
            return {}
        
        results = {}
        
        for model_name, model_func in power_models.items():
            print(f"\n使用 {model_name} 进行攻击...")
            
            correlations = []
            max_correlations = []
            
            # 对每个密钥猜测计算相关系数
            for key_guess in range(key_range[0], key_range[1]):
                # 计算理论功耗
                theoretical_power = []
                for plaintext in self.plaintexts:
                    theoretical_power.append(model_func(plaintext, key_guess))
                
                # 计算与实际功耗的相关系数
                trace_correlations = []
                for sample_idx in range(self.sample_points):
                    actual_power = [trace[sample_idx] for trace in self.power_traces]
                    
                    try:
                        corr, _ = pearsonr(theoretical_power, actual_power)
                        if np.isnan(corr):
                            corr = 0.0
                        trace_correlations.append(abs(corr))
                    except:
                        trace_correlations.append(0.0)
                
                correlations.append(trace_correlations)
                max_correlations.append(max(trace_correlations))
                
                print(f"密钥 {key_guess}: 最大相关系数 = {max(trace_correlations):.6f}")
            
            # 找到最佳密钥猜测
            best_key = np.argmax(max_correlations) + key_range[0]
            best_correlation = max(max_correlations)
            
            results[model_name] = {
                'best_key': best_key,
                'best_correlation': best_correlation,
                'all_correlations': correlations,
                'max_correlations': max_correlations
            }
            
            print(f"{model_name} 结果:")
            print(f"  推荐密钥: {best_key}")
            print(f"  最大相关系数: {best_correlation:.6f}")
            print(f"  真实密钥 {self.target_key & ((1 << (key_range[1] - key_range[0])) - 1)} 的相关系数: {max_correlations[self.target_key & ((1 << (key_range[1] - key_range[0])) - 1)]:.6f}")
        
        return results
    
    def analyze_results(self, results: Dict[str, Dict]) -> None:
        """
        分析攻击结果
        
        Args:
            results: CPA攻击结果
        """
        print("\n=== 攻击结果分析 ===")
        
        for model_name, result in results.items():
            print(f"\n{model_name}:")
            print(f"  推荐密钥: {result['best_key']}")
            print(f"  最大相关系数: {result['best_correlation']:.6f}")
            
            # 检查是否成功
            target_bits = self.target_key & 7  # 低3位
            if result['best_key'] == target_bits:
                print(f"  ✓ 攻击成功! 正确识别密钥低3位")
            else:
                print(f"  ✗ 攻击失败. 真实密钥低3位: {target_bits}")
    
    def visualize_results(self, results: Dict[str, Dict], save_path: str = None) -> None:
        """
        可视化攻击结果
        
        Args:
            results: CPA攻击结果
            save_path: 保存路径
        """
        print("\n=== 生成结果图表 ===")
        
        fig, axes = plt.subplots(len(results), 2, figsize=(15, 5*len(results)))
        if len(results) == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (model_name, result) in enumerate(results.items()):
            # 相关系数对比图
            ax1 = axes[idx, 0]
            keys = list(range(len(result['max_correlations'])))
            ax1.bar(keys, result['max_correlations'], alpha=0.7)
            ax1.set_title(f'{model_name} - 密钥相关系数对比')
            ax1.set_xlabel('密钥猜测值')
            ax1.set_ylabel('最大相关系数')
            ax1.grid(True, alpha=0.3)
            
            # 标记最佳密钥和真实密钥
            best_key = result['best_key']
            target_key = self.target_key & 7
            ax1.bar(best_key, result['max_correlations'][best_key], 
                   color='red', alpha=0.8, label=f'推荐密钥: {best_key}')
            if target_key != best_key:
                ax1.bar(target_key, result['max_correlations'][target_key], 
                       color='green', alpha=0.8, label=f'真实密钥: {target_key}')
            ax1.legend()
            
            # 相关系数迹线图
            ax2 = axes[idx, 1]
            correlations = np.array(result['all_correlations'])
            for key_idx in range(len(correlations)):
                if key_idx == best_key:
                    ax2.plot(correlations[key_idx], label=f'密钥 {key_idx} (推荐)', 
                            linewidth=2, color='red')
                elif key_idx == target_key:
                    ax2.plot(correlations[key_idx], label=f'密钥 {key_idx} (真实)', 
                            linewidth=2, color='green')
                else:
                    ax2.plot(correlations[key_idx], alpha=0.3, color='gray')
            
            ax2.set_title(f'{model_name} - 相关系数迹线')
            ax2.set_xlabel('采样点')
            ax2.set_ylabel('相关系数')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"结果图表已保存到: {save_path}")
        
        plt.show()
    
    def template_attack_demo(self) -> None:
        """
        演示模板攻击方法 (基于find_max.py的思路)
        """
        print("\n=== 模板攻击演示 ===")
        
        # 模拟PWM_Pipeline_Data的处理方式
        line_number = 600  # 每组的行数
        
        # 将功耗迹线按组处理
        grouped_data = []
        for i in range(0, len(self.power_traces), line_number):
            group = self.power_traces[i:i+line_number]
            if len(group) == line_number:
                # 找到每组的最大值
                max_values = []
                for trace in group:
                    max_values.append(np.max(trace))
                grouped_data.append(max_values)
        
        print(f"分组处理: {len(grouped_data)} 组数据，每组 {line_number} 条迹线")
        
        # 计算相关性 (简化版本)
        if len(grouped_data) >= 2:
            group1 = grouped_data[0]
            group2 = grouped_data[1]
            
            correlation = np.corrcoef(group1, group2)[0, 1]
            print(f"组间相关系数: {correlation:.6f}")
        
        return grouped_data

def main():
    """
    主函数 - 演示完整的CPA攻击流程
    """
    print("模乘模块CPA攻击演示")
    print("=" * 50)
    
    # 初始化CPA攻击
    power_file = '../data/ntt_pipeline_traces_x10k-3rd.txt'
    cpa = ModularMultiplicationCPA(
        power_traces_file=power_file,
        target_key=1000,
        sample_points=5000,
        max_traces=5000  # 限制迹线数量以加快演示
    )
    
    # 加载功耗迹线
    if not cpa.load_power_traces():
        print("无法加载功耗迹线，退出演示")
        return
    
    # 定义功耗模型
    power_models = {
        '汉明重量模型': cpa.hamming_weight_model,
        '位翻转模型': cpa.bit_flip_model,
        'NTT特定模型': cpa.ntt_specific_model
    }
    
    # 执行CPA攻击 (攻击低3位)
    results = cpa.perform_cpa_attack(power_models, key_range=(0, 8))
    
    # 分析结果
    cpa.analyze_results(results)
    
    # 可视化结果
    cpa.visualize_results(results, 'modular_multiplication_cpa_results.png')
    
    # 演示模板攻击
    cpa.template_attack_demo()
    
    print("\n=== 演示完成 ===")
    print("\n总结:")
    print("1. 功耗曲线格式: 明文:功耗迹线 (冒号分隔)")
    print("2. 硬件对应: NTT流水线模乘运算的功耗特征")
    print("3. 攻击方法: 相关性功耗分析 (CPA)")
    print("4. 成功关键: 精确的功耗模型和足够的统计样本")

if __name__ == '__main__':
    main()