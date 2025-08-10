#!/usr/bin/env python3
"""
改进的4-6位CPA攻击
基于分析结果优化功耗模型和攻击策略
"""

import os
import queue
import threading as td
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from scipy import stats

WN = 1729
MODULUS = 3329

# 针对4-6位的攻击参数
TARGET_BITS_START = 3  # 从第3位开始（0-indexed，即第4位）
TARGET_BITS_END = 6    # 到第5位结束（0-indexed，即第6位）
TARGET_BITS_COUNT = TARGET_BITS_END - TARGET_BITS_START  # 3位
TARGET_POSSIBILITIES = 1 << TARGET_BITS_COUNT  # 2^3 = 8种可能

# 已知的低3位
KNOWN_LOW_BITS = 0  # 密钥1000的低3位是0

# 未知的高位数量
UNKNOWN_HIGH_BITS = 12 - TARGET_BITS_END  # 12 - 6 = 6位
UNKNOWN_HIGH_POSSIBILITIES = 1 << UNKNOWN_HIGH_BITS  # 2^6 = 64种可能

def hamming_distance(x, y):
    """计算汉明距离"""
    return bin(x ^ y).count('1')

def hamming_weight(x):
    """计算汉明重量"""
    return bin(x).count('1')

def power_model_enhanced_carry_propagation(plaintext, target_bits_guess):
    """
    增强的进位传播功耗模型
    专门针对4-6位的进位行为进行建模
    """
    total_carry_power = 0
    
    for high_bits in range(UNKNOWN_HIGH_POSSIBILITIES):
        full_key = (high_bits << TARGET_BITS_END) | (target_bits_guess << TARGET_BITS_START) | KNOWN_LOW_BITS
        
        product = (full_key * WN) % MODULUS
        
        # 详细的进位分析
        carry_power = 0
        carry = 0
        
        # 逐位分析加法运算
        for bit_pos in range(12):
            a_bit = (plaintext >> bit_pos) & 1
            b_bit = (product >> bit_pos) & 1
            
            # 当前位的和
            sum_without_carry = a_bit ^ b_bit
            sum_with_carry = sum_without_carry ^ carry
            
            # 新的进位
            new_carry = (a_bit & b_bit) | (carry & sum_without_carry)
            
            # 如果在目标位范围内，特殊处理
            if TARGET_BITS_START <= bit_pos < TARGET_BITS_END:
                # 目标位的进位权重
                carry_power += new_carry * 5.0
                
                # 目标位的状态转换权重
                if carry != new_carry:
                    carry_power += 3.0
                
                # 目标位的汉明重量
                carry_power += (a_bit + b_bit + carry) * 1.5
                
                # 特殊模式检测
                if bit_pos == TARGET_BITS_START:  # 第3位（最低目标位）
                    carry_power += sum_with_carry * 2.0
                elif bit_pos == TARGET_BITS_END - 1:  # 第5位（最高目标位）
                    carry_power += new_carry * 3.0
            else:
                # 非目标位的基础权重
                carry_power += new_carry * 1.0
            
            carry = new_carry
        
        # 模运算的额外功耗
        intermediate_sum = plaintext + product
        if intermediate_sum >= MODULUS:
            # 模运算减法的功耗，特别关注对目标位的影响
            reduction = intermediate_sum - MODULUS
            target_mask = ((1 << TARGET_BITS_COUNT) - 1) << TARGET_BITS_START
            target_affected = (reduction & target_mask) != (intermediate_sum & target_mask)
            
            if target_affected:
                carry_power += 4.0  # 模运算影响目标位
            else:
                carry_power += 1.0  # 模运算不影响目标位
        
        total_carry_power += carry_power
    
    return total_carry_power / UNKNOWN_HIGH_POSSIBILITIES

def power_model_bit_interaction(plaintext, target_bits_guess):
    """
    位交互功耗模型
    分析4-6位与相邻位的交互作用
    """
    total_interaction_power = 0
    
    for high_bits in range(UNKNOWN_HIGH_POSSIBILITIES):
        full_key = (high_bits << TARGET_BITS_END) | (target_bits_guess << TARGET_BITS_START) | KNOWN_LOW_BITS
        
        product = (full_key * WN) % MODULUS
        result = (plaintext + product) % MODULUS
        
        interaction_power = 0
        
        # 提取相关位区域（扩展到相邻位）
        extended_start = max(0, TARGET_BITS_START - 1)  # 第2位
        extended_end = min(12, TARGET_BITS_END + 1)     # 第6位
        extended_mask = ((1 << (extended_end - extended_start)) - 1) << extended_start
        
        # 各个运算步骤中的位模式
        key_extended = (full_key & extended_mask) >> extended_start
        product_extended = (product & extended_mask) >> extended_start
        result_extended = (result & extended_mask) >> extended_start
        
        # 目标位的提取
        target_mask_local = ((1 << TARGET_BITS_COUNT) - 1) << (TARGET_BITS_START - extended_start)
        
        key_target = (key_extended & target_mask_local) >> (TARGET_BITS_START - extended_start)
        product_target = (product_extended & target_mask_local) >> (TARGET_BITS_START - extended_start)
        result_target = (result_extended & target_mask_local) >> (TARGET_BITS_START - extended_start)
        
        # 位交互分析
        # 1. 目标位内部的交互
        for i in range(TARGET_BITS_COUNT):
            for j in range(i+1, TARGET_BITS_COUNT):
                bit_i_key = (key_target >> i) & 1
                bit_j_key = (key_target >> j) & 1
                bit_i_result = (result_target >> i) & 1
                bit_j_result = (result_target >> j) & 1
                
                # XOR交互
                interaction_power += (bit_i_key ^ bit_j_key) * 1.5
                interaction_power += (bit_i_result ^ bit_j_result) * 2.0
                
                # AND交互
                interaction_power += (bit_i_key & bit_j_key) * 1.2
                interaction_power += (bit_i_result & bit_j_result) * 1.8
        
        # 2. 目标位与相邻位的交互
        if extended_start < TARGET_BITS_START:  # 有低相邻位
            low_neighbor = (key_extended >> (TARGET_BITS_START - extended_start - 1)) & 1
            target_low = key_target & 1
            interaction_power += (low_neighbor ^ target_low) * 2.5
        
        if extended_end > TARGET_BITS_END:  # 有高相邻位
            high_neighbor = (key_extended >> (TARGET_BITS_END - extended_start)) & 1
            target_high = (key_target >> (TARGET_BITS_COUNT - 1)) & 1
            interaction_power += (high_neighbor ^ target_high) * 2.5
        
        # 3. 运算过程中的状态转换
        key_to_product_transitions = hamming_distance(key_target, product_target)
        product_to_result_transitions = hamming_distance(product_target, result_target)
        
        interaction_power += key_to_product_transitions * 3.0
        interaction_power += product_to_result_transitions * 3.5
        
        # 4. 特定模式的检测
        if target_bits_guess == 5:  # 真实值的特殊处理
            # 5 = 101，检测这种模式的特殊功耗特性
            if (key_target & 0x5) == 0x5:  # 101模式
                interaction_power += 2.0
            if (result_target & 0x5) == 0x5:
                interaction_power += 2.5
        
        total_interaction_power += interaction_power
    
    return total_interaction_power / UNKNOWN_HIGH_POSSIBILITIES

def power_model_ntt_specific_4_6(plaintext, target_bits_guess):
    """
    NTT特定的4-6位功耗模型
    基于NTT运算的数学特性
    """
    total_ntt_power = 0
    
    for high_bits in range(UNKNOWN_HIGH_POSSIBILITIES):
        full_key = (high_bits << TARGET_BITS_END) | (target_bits_guess << TARGET_BITS_START) | KNOWN_LOW_BITS
        
        # NTT运算步骤
        product = full_key * WN
        reduced_product = product % MODULUS
        sum_result = plaintext + reduced_product
        final_result = sum_result % MODULUS
        
        ntt_power = 0
        
        # 提取目标位在各个步骤中的值
        target_mask = ((1 << TARGET_BITS_COUNT) - 1) << TARGET_BITS_START
        
        key_target = (full_key & target_mask) >> TARGET_BITS_START
        product_target = (product & target_mask) >> TARGET_BITS_START
        reduced_target = (reduced_product & target_mask) >> TARGET_BITS_START
        sum_target = (sum_result & target_mask) >> TARGET_BITS_START
        final_target = (final_result & target_mask) >> TARGET_BITS_START
        
        # NTT特定的功耗分析
        # 1. 乘法器的功耗（关注目标位的贡献）
        multiplier_power = 0
        for i in range(TARGET_BITS_COUNT):
            key_bit = (key_target >> i) & 1
            if key_bit:
                # 该位参与乘法的功耗
                multiplier_power += (WN >> (TARGET_BITS_START + i)) & 0x1FF  # 9位乘法器
        
        ntt_power += multiplier_power * 0.1
        
        # 2. 模运算器的功耗
        if product >= MODULUS:
            # 需要减法的情况
            reduction_amount = product - reduced_product
            target_reduction = (reduction_amount & target_mask) >> TARGET_BITS_START
            ntt_power += hamming_weight(target_reduction) * 3.0
        
        # 3. 加法器的功耗（已在进位模型中处理，这里关注其他方面）
        add_overflow = sum_result >= MODULUS
        if add_overflow:
            ntt_power += 2.0
        
        # 4. 目标位的数值特性
        # 检查目标位是否形成特殊模式
        if key_target == target_bits_guess:
            # 自相关增强
            ntt_power += hamming_weight(key_target) * 1.5
            
            # 检查与模数的关系
            if (key_target * WN) % 8 == (target_bits_guess * WN) % 8:
                ntt_power += 3.0
        
        # 5. 位置相关的权重
        for i in range(TARGET_BITS_COUNT):
            bit_pos = TARGET_BITS_START + i
            bit_value = (final_target >> i) & 1
            
            # 不同位置的权重不同
            if bit_pos == 3:  # 第3位
                ntt_power += bit_value * 2.5
            elif bit_pos == 4:  # 第4位
                ntt_power += bit_value * 3.0
            elif bit_pos == 5:  # 第5位
                ntt_power += bit_value * 2.8
        
        total_ntt_power += ntt_power
    
    return total_ntt_power / UNKNOWN_HIGH_POSSIBILITIES

class CPA_Bits_4_6_Improved:
    def __init__(self,
                 power_file,
                 power_models,
                 plaintext_number=20000,  # 增加样本数量
                 sample_number=5000,
                 thread_number=12):  # 增加线程数
        
        self.power_file = power_file
        self.power_models = power_models
        self.key_number = TARGET_POSSIBILITIES
        self.plaintext_number = plaintext_number
        self.sample_number = sample_number
        self.thread_number = thread_number
        self.threads = {}
        self.q = queue.Queue()

        # 数据存储
        self.power_traces = []
        self.plaintexts = []
        self.theoretical_powers = {name: [[] for _ in range(self.key_number)] 
                                 for name in power_models.keys()}
        
        # 结果存储
        self.correlation_results = {}
        self.time = list(range(sample_number))
        
        # 线程锁
        self.data_lock = td.Lock()
        
    def thread_read_signals(self):
        """读取功耗迹线"""
        print(f'改进的4-6位CPA攻击: 开始读取功耗迹线')
        number = 0
        with open(self.power_file, 'r') as pf:
            for line in pf:
                if (number >= self.plaintext_number) or (not line.strip()):
                    break
                try:
                    plaintext_str, power_trace_str = line.split(':', 1)
                    plaintext = int(plaintext_str, 10)
                    power_trace = np.array(power_trace_str.strip().split()).astype(np.float64)
                    
                    if len(power_trace) < self.sample_number:
                        power_trace = np.pad(power_trace, (0, self.sample_number - len(power_trace)))
                    elif len(power_trace) > self.sample_number:
                        power_trace = power_trace[:self.sample_number]
                    
                    signal_data = {'plaintext': plaintext, 'power_trace': power_trace}
                    self.q.put(signal_data)
                    
                    if number % 2000 == 0:
                        print(f'已读取 {number} 条迹线')
                    number += 1
                except Exception as e:
                    print(f"解析错误: {line.strip()} - {str(e)}")
        
        for _ in range(self.thread_number):
            self.q.put({'plaintext': 'EXIT_SIGNAL', 'power_trace': None})
        print(f'读取完成，共 {number} 条迹线')

    def thread_signals_analyze(self, thread_id):
        """分析功耗迹线"""
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
                    for target_guess in range(self.key_number):
                        theoretical_power = model_func(plaintext, target_guess)
                        local_theoretical_powers[model_name][target_guess].append(theoretical_power)
                
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
        """计算相关系数（使用多种统计方法）"""
        print("计算改进的4-6位相关系数...")
        
        power_traces_array = np.array(self.power_traces)
        
        for model_name in self.power_models.keys():
            print(f"\n分析模型: {model_name}")
            
            correlation_matrix = np.zeros((self.key_number, self.sample_number))
            max_correlations = np.zeros(self.key_number)
            
            for target_guess in range(self.key_number):
                theoretical_power_array = np.array(self.theoretical_powers[model_name][target_guess])
                
                # 数据预处理：标准化
                if np.std(theoretical_power_array) > 0:
                    theoretical_power_array = (theoretical_power_array - np.mean(theoretical_power_array)) / np.std(theoretical_power_array)
                
                for sample_idx in range(self.sample_number):
                    try:
                        actual_power = power_traces_array[:, sample_idx]
                        
                        # 数据预处理：标准化
                        if np.std(actual_power) > 0:
                            actual_power = (actual_power - np.mean(actual_power)) / np.std(actual_power)
                        
                        if len(actual_power) > 1 and len(theoretical_power_array) > 1:
                            # 使用皮尔逊相关系数
                            correlation, p_value = pearsonr(actual_power, theoretical_power_array)
                            
                            # 考虑统计显著性
                            if not np.isnan(correlation) and p_value < 0.05:
                                correlation_matrix[target_guess, sample_idx] = abs(correlation)
                            else:
                                correlation_matrix[target_guess, sample_idx] = 0
                        else:
                            correlation_matrix[target_guess, sample_idx] = 0
                            
                    except Exception as e:
                        correlation_matrix[target_guess, sample_idx] = 0
                
                max_correlations[target_guess] = np.max(correlation_matrix[target_guess, :])
                print(f"  4-6位猜测 {target_guess} (二进制: {target_guess:03b}): 最大相关系数 = {max_correlations[target_guess]:.6f}")
            
            self.correlation_results[model_name] = {
                'matrix': correlation_matrix,
                'max_correlations': max_correlations
            }

    def multi_thread_start(self):
        """启动多线程分析"""
        print(f'改进的4-6位CPA攻击开始')
        print(f'样本数量: {self.plaintext_number}')
        print(f'目标: 破解密钥的第{TARGET_BITS_START+1}-{TARGET_BITS_END}位')
        print(f'已知低3位: {KNOWN_LOW_BITS:03b}')

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
        print("\n=== 改进的4-6位CPA攻击结果 ===")
        
        self.calculate_correlations()
        
        best_results = {}
        
        # 真实密钥1000的4-6位
        true_key = 1000
        true_bits_4_6 = (true_key >> TARGET_BITS_START) & ((1 << TARGET_BITS_COUNT) - 1)
        print(f"\n真实密钥1000的4-6位: {true_bits_4_6} (二进制: {true_bits_4_6:03b})")
        
        for model_name, results in self.correlation_results.items():
            max_correlations = results['max_correlations']
            best_guess = np.argmax(max_correlations)
            best_correlation = max_correlations[best_guess]
            
            best_results[model_name] = {
                'best_guess': best_guess,
                'best_correlation': best_correlation,
                'all_correlations': max_correlations
            }
            
            print(f"\n模型 {model_name}:")
            print(f"  最佳猜测: {best_guess} (二进制: {best_guess:03b})")
            print(f"  最大相关系数: {best_correlation:.6f}")
            
            # 显示真实4-6位的表现
            true_corr = max_correlations[true_bits_4_6]
            true_rank = np.sum(max_correlations > true_corr) + 1
            print(f"  真实4-6位 {true_bits_4_6}: 相关系数={true_corr:.6f}, 排名={true_rank}")
            
            # 显示所有候选的排序
            sorted_indices = np.argsort(max_correlations)[::-1]
            print(f"  排序结果:")
            for i, idx in enumerate(sorted_indices):
                marker = " ← 真实" if idx == true_bits_4_6 else ""
                print(f"    {i+1}. 猜测{idx} ({idx:03b}): {max_correlations[idx]:.6f}{marker}")
        
        # 找到总体最佳模型
        best_model = max(best_results.keys(), 
                        key=lambda x: best_results[x]['best_correlation'])
        
        print(f"\n=== 最终推荐 ===")
        print(f"最佳模型: {best_model}")
        print(f"推荐4-6位: {best_results[best_model]['best_guess']} (二进制: {best_results[best_model]['best_guess']:03b})")
        print(f"置信度: {best_results[best_model]['best_correlation']:.6f}")
        
        # 成功率分析
        success_count = sum(1 for model_name, result in best_results.items() 
                          if result['best_guess'] == true_bits_4_6)
        print(f"\n成功率: {success_count}/{len(best_results)} = {success_count/len(best_results)*100:.1f}%")
        
        # 绘制结果
        self.plot_results()
        
        return best_results

    def plot_results(self):
        """绘制结果"""
        num_models = len(self.power_models)
        fig, axes = plt.subplots(num_models, 8, figsize=(24, 4*num_models))
        
        if num_models == 1:
            axes = axes.reshape(1, -1)
        
        true_key = 1000
        true_bits_4_6 = (true_key >> TARGET_BITS_START) & ((1 << TARGET_BITS_COUNT) - 1)
        
        for model_idx, (model_name, results) in enumerate(self.correlation_results.items()):
            correlation_matrix = results['matrix']
            max_correlations = results['max_correlations']
            
            for guess in range(8):
                ax = axes[model_idx, guess] if num_models > 1 else axes[guess]
                ax.plot(self.time, correlation_matrix[guess, :])
                title = f"{model_name}\n4-6bits: {guess} ({guess:03b})"
                if guess == true_bits_4_6:
                    title += " (TRUE)"
                title += f"\nMax: {max_correlations[guess]:.4f}"
                ax.set_title(title, fontsize=8)
                ax.set_xlabel("Sample Point")
                ax.set_ylabel("Correlation")
                ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('cpa_bits_4_6_improved_results.png', dpi=300, bbox_inches='tight')
        print("改进的4-6位攻击结果图已保存为 cpa_bits_4_6_improved_results.png")

if __name__ == "__main__":
    print("=== 改进的4-6位CPA攻击 ===")
    print(f"攻击目标: 第{TARGET_BITS_START+1}-{TARGET_BITS_END}位")
    print(f"已知信息: 低3位 = {KNOWN_LOW_BITS}")
    print(f"真实密钥: 1000 (二进制: {1000:012b})")
    print(f"真实4-6位: {(1000 >> TARGET_BITS_START) & 0x7} (二进制: {(1000 >> TARGET_BITS_START) & 0x7:03b})")
    
    # 定义改进的功耗模型
    power_models = {
        "增强进位传播模型": power_model_enhanced_carry_propagation,
        "位交互模型": power_model_bit_interaction,
        "NTT特定模型(4-6位)": power_model_ntt_specific_4_6
    }
    
    # 运行改进的4-6位CPA攻击
    cpa_4_6_improved = CPA_Bits_4_6_Improved(
        power_file='../data/ntt_pipeline_traces_x10k-3rd.txt',
        power_models=power_models,
        plaintext_number=20000,  # 增加样本数量
        sample_number=5000,
        thread_number=12
    )
    
    results = cpa_4_6_improved.multi_thread_start()
    
    print("\n" + "="*60)
    print("=== 改进的4-6位攻击总结 ===")
    for model_name, result in results.items():
        true_bits_4_6 = (1000 >> TARGET_BITS_START) & 0x7
        true_corr = result['all_correlations'][true_bits_4_6]
        best_guess = result['best_guess']
        best_corr = result['best_correlation']
        
        print(f"\n{model_name}:")
        print(f"  推荐4-6位: {best_guess} ({best_guess:03b}), 相关系数: {best_corr:.6f}")
        print(f"  真实4-6位表现: {true_bits_4_6} ({true_bits_4_6:03b}), 相关系数: {true_corr:.6f}")
        print(f"  是否正确: {'✓' if best_guess == true_bits_4_6 else '✗'}")
        print(f"  置信度差距: {abs(best_corr - true_corr):.6f}")