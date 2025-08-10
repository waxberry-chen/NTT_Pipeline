#!/usr/bin/env python3
"""
改进的CPA攻击实现
针对NTT运算的特定功耗特性进行优化
"""

import os
import queue
import threading as td
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

WN = 1729
MODULUS = 3329
NUM_UNKNOWN_BITS = 9
UNKNOWN_POSSIBILITIES = 1 << NUM_UNKNOWN_BITS

def hamming_weight(x):
    """计算汉明重量"""
    return bin(x).count('1')

def hamming_distance(x, y):
    """计算汉明距离"""
    return bin(x ^ y).count('1')

def power_model_ntt_specific(plaintext, key_guess):
    """
    NTT特定的功耗模型
    考虑模运算、乘法运算和加法运算的功耗特性
    """
    total_power = 0
    
    for b_high in range(UNKNOWN_POSSIBILITIES):
        # 重构完整密钥
        b_full_guess = (b_high << 3) | key_guess
        
        # NTT运算步骤的功耗建模
        # 1. 乘法运算: b * WN
        mult_result = b_full_guess * WN
        mult_power = hamming_weight(mult_result)
        
        # 2. 模运算: (b * WN) % MODULUS
        mod_result = mult_result % MODULUS
        mod_power = hamming_weight(mod_result)
        
        # 3. 加法运算: a + (b * WN) % MODULUS
        add_result = plaintext + mod_result
        add_power = hamming_weight(add_result)
        
        # 4. 最终模运算: (a + (b * WN) % MODULUS) % MODULUS
        final_result = add_result % MODULUS
        final_power = hamming_weight(final_result)
        
        # 5. 考虑进位和借位的功耗
        carry_power = hamming_distance(add_result, final_result)
        
        # 综合功耗模型（加权组合）
        step_power = (
            0.2 * mult_power +     # 乘法功耗
            0.3 * mod_power +      # 第一次模运算功耗
            0.2 * add_power +      # 加法功耗
            0.2 * final_power +    # 最终结果功耗
            0.1 * carry_power      # 进位/借位功耗
        )
        
        total_power += step_power
    
    return total_power / UNKNOWN_POSSIBILITIES

def power_model_conditional_reduction(plaintext, key_guess):
    """
    基于条件约简的功耗模型
    重点关注是否发生模运算约简的功耗差异
    """
    reduction_count = 0
    total_intermediate_hw = 0
    
    for b_high in range(UNKNOWN_POSSIBILITIES):
        b_full_guess = (b_high << 3) | key_guess
        
        # 计算中间结果
        product = (b_full_guess * WN) % MODULUS
        intermediate = plaintext + product
        
        # 检查是否需要模运算约简
        if intermediate >= MODULUS:
            reduction_count += 1
            # 约简操作的功耗
            reduced_value = intermediate - MODULUS
            power = hamming_weight(reduced_value) + hamming_distance(intermediate, reduced_value)
        else:
            # 无约简的功耗
            power = hamming_weight(intermediate)
        
        total_intermediate_hw += power
    
    # 返回平均功耗，同时考虑约简概率
    avg_power = total_intermediate_hw / UNKNOWN_POSSIBILITIES
    reduction_prob = reduction_count / UNKNOWN_POSSIBILITIES
    
    # 约简概率作为额外的功耗特征
    return avg_power + 2.0 * reduction_prob  # 约简操作消耗更多功耗

def power_model_bit_transitions(plaintext, key_guess):
    """
    基于位转换的功耗模型
    关注数据路径中的位翻转
    """
    total_transitions = 0
    
    for b_high in range(UNKNOWN_POSSIBILITIES):
        b_full_guess = (b_high << 3) | key_guess
        
        # 计算各个中间值
        step1 = b_full_guess * WN
        step2 = step1 % MODULUS
        step3 = plaintext + step2
        step4 = step3 % MODULUS
        
        # 计算每一步的位转换
        transitions = 0
        
        # 输入到乘法器的转换
        transitions += hamming_distance(b_full_guess, plaintext)
        
        # 乘法结果的位转换
        transitions += hamming_distance(step1, step2)
        
        # 加法器的位转换
        transitions += hamming_distance(step2, step3)
        
        # 最终模运算的位转换
        transitions += hamming_distance(step3, step4)
        
        total_transitions += transitions
    
    return total_transitions / UNKNOWN_POSSIBILITIES

class ImprovedCPA:
    def __init__(self,
                 power_file,
                 power_models,  # 支持多个功耗模型
                 key_number,
                 plaintext_number=4096,
                 sample_number=5000,
                 thread_number=10):
        
        self.power_file = power_file
        self.power_models = power_models  # 字典格式: {"model_name": model_function}
        self.key_number = key_number
        self.plaintext_number = plaintext_number
        self.sample_number = sample_number
        self.thread_number = thread_number
        self.threads = {}
        self.q = queue.Queue()

        # 数据存储
        self.power_traces = []
        self.plaintexts = []
        self.theoretical_powers = {name: [[] for _ in range(key_number)] 
                                 for name in power_models.keys()}
        
        # 结果存储
        self.correlation_results = {}
        self.time = list(range(sample_number))
        
        # 线程锁
        self.data_lock = td.Lock()
        
    def thread_read_signals(self):
        """读取功耗迹线"""
        print(f'Improved CPA: 开始读取功耗迹线')
        number = 0
        with open(self.power_file, 'r') as pf:
            for line in pf:
                if (number >= self.plaintext_number) or (not line.strip()):
                    break
                try:
                    plaintext_str, power_trace_str = line.split(':', 1)
                    plaintext = int(plaintext_str, 10)  # 十进制明文
                    power_trace = np.array(power_trace_str.strip().split()).astype(np.float64)
                    
                    if len(power_trace) < self.sample_number:
                        power_trace = np.pad(power_trace, (0, self.sample_number - len(power_trace)))
                    elif len(power_trace) > self.sample_number:
                        power_trace = power_trace[:self.sample_number]
                    
                    signal_data = {'plaintext': plaintext, 'power_trace': power_trace}
                    self.q.put(signal_data)
                    
                    if number % 1000 == 0:
                        print(f'已读取 {number} 条迹线')
                    number += 1
                except Exception as e:
                    print(f"解析错误: {line.strip()} - {str(e)}")
        
        for _ in range(self.thread_number):
            self.q.put({'plaintext': 'EXIT_SIGNAL', 'power_trace': None})
        print(f'读取完成，共 {number} 条迹线')

    def thread_signals_analyze(self, thread_id):
        """分析功耗迹线"""
        print(f'分析线程 {thread_id} 启动')
        processed_count = 0
        local_power_traces = []
        local_plaintexts = []
        local_theoretical_powers = {name: [[] for _ in range(self.key_number)] 
                                  for name in self.power_models.keys()}
        
        while True:
            try:
                signal = self.q.get(timeout=10)
                if signal['plaintext'] == 'EXIT_SIGNAL':
                    self.q.task_done()
                    break
                
                plaintext = signal['plaintext']
                power_trace = signal['power_trace']
                
                local_power_traces.append(power_trace)
                local_plaintexts.append(plaintext)
                
                # 计算所有功耗模型的理论功耗
                for model_name, model_func in self.power_models.items():
                    for key in range(self.key_number):
                        theoretical_power = model_func(plaintext, key)
                        local_theoretical_powers[model_name][key].append(theoretical_power)
                
                processed_count += 1
                self.q.task_done()
                
            except queue.Empty:
                break
            except Exception as e:
                print(f"线程 {thread_id} 错误: {str(e)}")
        
        # 合并数据
        with self.data_lock:
            self.power_traces.extend(local_power_traces)
            self.plaintexts.extend(local_plaintexts)
            for model_name in self.power_models.keys():
                for key in range(self.key_number):
                    self.theoretical_powers[model_name][key].extend(
                        local_theoretical_powers[model_name][key])
        
        print(f'线程 {thread_id} 完成，处理 {processed_count} 条迹线')

    def calculate_correlations(self):
        """计算所有模型的相关系数"""
        print("计算相关系数...")
        
        power_traces_array = np.array(self.power_traces)
        
        for model_name in self.power_models.keys():
            print(f"\n分析模型: {model_name}")
            
            correlation_matrix = np.zeros((self.key_number, self.sample_number))
            max_correlations = np.zeros(self.key_number)
            
            for key in range(self.key_number):
                theoretical_power_array = np.array(self.theoretical_powers[model_name][key])
                
                for sample_idx in range(self.sample_number):
                    try:
                        actual_power = power_traces_array[:, sample_idx]
                        
                        if len(actual_power) > 1 and len(theoretical_power_array) > 1:
                            correlation, _ = pearsonr(actual_power, theoretical_power_array)
                            if not np.isnan(correlation):
                                correlation_matrix[key, sample_idx] = abs(correlation)
                            else:
                                correlation_matrix[key, sample_idx] = 0
                        else:
                            correlation_matrix[key, sample_idx] = 0
                            
                    except Exception as e:
                        correlation_matrix[key, sample_idx] = 0
                
                max_correlations[key] = np.max(correlation_matrix[key, :])
                print(f"  Key {key}: 最大相关系数 = {max_correlations[key]:.6f}")
            
            self.correlation_results[model_name] = {
                'matrix': correlation_matrix,
                'max_correlations': max_correlations
            }

    def multi_thread_start(self):
        """启动多线程分析"""
        print(f'改进CPA攻击开始')

        reader = td.Thread(target=self.thread_read_signals)
        reader.start()
        self.threads['reader'] = reader

        analyzers = []
        for thread_id in range(self.thread_number):
            analyzer = td.Thread(target=self.thread_signals_analyze, args=(thread_id,))
            analyzer.daemon = True
            analyzer.start()
            analyzers.append(analyzer)
            self.threads[thread_id] = analyzer

        reader.join()
        print("读取线程完成")
        self.q.join()
        print("所有队列任务完成")

        for analyzer in analyzers:
            analyzer.join(timeout=5)

        print("\n所有线程完成")
        return self.result_analyze()

    def result_analyze(self):
        """分析结果"""
        print("\n=== 改进CPA分析结果 ===")
        
        self.calculate_correlations()
        
        best_results = {}
        
        for model_name, results in self.correlation_results.items():
            max_correlations = results['max_correlations']
            best_key = np.argmax(max_correlations)
            best_correlation = max_correlations[best_key]
            
            best_results[model_name] = {
                'best_key': best_key,
                'best_correlation': best_correlation,
                'all_correlations': max_correlations
            }
            
            print(f"\n模型 {model_name}:")
            print(f"  最佳密钥: {best_key} (0x{best_key:X})")
            print(f"  最大相关系数: {best_correlation:.6f}")
            
            # 显示真实密钥(Key 0)的表现
            true_key_corr = max_correlations[0]
            true_key_rank = np.sum(max_correlations > true_key_corr) + 1
            print(f"  真实密钥Key 0: 相关系数={true_key_corr:.6f}, 排名={true_key_rank}")
        
        # 找到总体最佳模型
        best_model = max(best_results.keys(), 
                        key=lambda x: best_results[x]['best_correlation'])
        
        print(f"\n=== 最终推荐 ===")
        print(f"最佳模型: {best_model}")
        print(f"推荐密钥: {best_results[best_model]['best_key']} (0x{best_results[best_model]['best_key']:X})")
        print(f"置信度: {best_results[best_model]['best_correlation']:.6f}")
        
        # 绘制结果
        self.plot_results()
        
        return best_results

    def plot_results(self):
        """绘制所有模型的结果"""
        num_models = len(self.power_models)
        fig, axes = plt.subplots(num_models, 8, figsize=(20, 4*num_models))
        
        if num_models == 1:
            axes = axes.reshape(1, -1)
        
        for model_idx, (model_name, results) in enumerate(self.correlation_results.items()):
            correlation_matrix = results['matrix']
            max_correlations = results['max_correlations']
            
            for key in range(self.key_number):
                ax = axes[model_idx, key]
                ax.plot(self.time, correlation_matrix[key, :])
                title = f"{model_name}\nKey {key}"
                if key == 0:
                    title += " (真实)"
                title += f"\nMax: {max_correlations[key]:.4f}"
                ax.set_title(title, fontsize=8)
                ax.set_xlabel("Sample Point")
                ax.set_ylabel("Correlation")
                ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('improved_cpa_results.png', dpi=300, bbox_inches='tight')
        print("结果图已保存为 improved_cpa_results.png")

if __name__ == "__main__":
    print("=== 改进的CPA攻击 ===")
    
    # 定义多个功耗模型
    power_models = {
        # "NTT特定模型": power_model_ntt_specific,
        # "条件约简模型": power_model_conditional_reduction,
        "位转换模型": power_model_bit_transitions
    }
    
    # 运行改进的CPA攻击
    improved_cpa = ImprovedCPA(
        power_file='../data/ntt_pipeline_traces_x10k-3rd.txt',
        power_models=power_models,
        key_number=8,
        plaintext_number=10000,
        sample_number=5000,
        thread_number=10
    )
    
    results = improved_cpa.multi_thread_start()
    
    print("\n" + "="*60)
    print("=== 模型对比总结 ===")
    for model_name, result in results.items():
        true_key_corr = result['all_correlations'][0]
        best_key = result['best_key']
        best_corr = result['best_correlation']
        
        print(f"\n{model_name}:")
        print(f"  推荐密钥: {best_key}, 相关系数: {best_corr:.6f}")
        print(f"  真实密钥表现: 相关系数: {true_key_corr:.6f}")
        print(f"  是否正确: {'✓' if best_key == 0 else '✗'}")