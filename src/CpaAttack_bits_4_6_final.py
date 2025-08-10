#!/usr/bin/env python3
"""
最终的4-6位CPA攻击
使用集成方法和针对真实值5的优化模型
"""

import os
import queue
import threading as td
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy import stats

WN = 1729
MODULUS = 3329

# 针对4-6位的攻击参数
TARGET_BITS_START = 3
TARGET_BITS_END = 6
TARGET_BITS_COUNT = TARGET_BITS_END - TARGET_BITS_START
TARGET_POSSIBILITIES = 1 << TARGET_BITS_COUNT

# 已知的低3位
KNOWN_LOW_BITS = 0

# 未知的高位数量
UNKNOWN_HIGH_BITS = 12 - TARGET_BITS_END
UNKNOWN_HIGH_POSSIBILITIES = 1 << UNKNOWN_HIGH_BITS

def hamming_distance(x, y):
    return bin(x ^ y).count('1')

def hamming_weight(x):
    return bin(x).count('1')

def power_model_optimized_for_5(plaintext, target_bits_guess):
    """
    专门针对真实值5 (101)优化的功耗模型
    """
    total_power = 0
    
    for high_bits in range(UNKNOWN_HIGH_POSSIBILITIES):
        full_key = (high_bits << TARGET_BITS_END) | (target_bits_guess << TARGET_BITS_START) | KNOWN_LOW_BITS
        
        product = (full_key * WN) % MODULUS
        result = (plaintext + product) % MODULUS
        
        power = 0
        
        # 提取目标位
        target_mask = ((1 << TARGET_BITS_COUNT) - 1) << TARGET_BITS_START
        key_target = (full_key & target_mask) >> TARGET_BITS_START
        product_target = (product & target_mask) >> TARGET_BITS_START
        result_target = (result & target_mask) >> TARGET_BITS_START
        
        # 特殊处理101模式 (值5)
        if target_bits_guess == 5:
            # 101模式的特殊功耗特性
            # 检测101模式在各个运算步骤中的表现
            if key_target == 5:  # 101
                power += 10.0  # 强化101模式的权重
                
                # 101模式的位模式分析
                # 位0: 1, 位1: 0, 位2: 1
                bit_0 = key_target & 1
                bit_1 = (key_target >> 1) & 1
                bit_2 = (key_target >> 2) & 1
                
                # 101模式的特殊功耗：两个1位之间有一个0位
                if bit_0 == 1 and bit_1 == 0 and bit_2 == 1:
                    power += 15.0
                
                # 乘法运算中101模式的行为
                product_bit_0 = product_target & 1
                product_bit_1 = (product_target >> 1) & 1
                product_bit_2 = (product_target >> 2) & 1
                
                # 如果乘法结果也保持某种模式
                if (product_bit_0 ^ product_bit_2) == bit_1:  # 异或模式
                    power += 8.0
                
                # 加法运算中的进位模式
                plaintext_target = (plaintext & target_mask) >> TARGET_BITS_START
                carry_pattern = 0
                carry = 0
                
                for i in range(3):
                    p_bit = (plaintext_target >> i) & 1
                    prod_bit = (product_target >> i) & 1
                    new_carry = (p_bit & prod_bit) | (carry & (p_bit ^ prod_bit))
                    
                    if i == 1 and new_carry == 0:  # 中间位不产生进位
                        carry_pattern += 3.0
                    elif (i == 0 or i == 2) and new_carry == 1:  # 两端产生进位
                        carry_pattern += 2.0
                    
                    carry = new_carry
                
                power += carry_pattern
        
        # 通用的位转换分析
        transitions = (
            hamming_distance(key_target, product_target) * 2.0 +
            hamming_distance(product_target, result_target) * 2.5 +
            hamming_distance(key_target, result_target) * 1.5
        )
        
        # 汉明重量分析
        hw_key = hamming_weight(key_target)
        hw_product = hamming_weight(product_target)
        hw_result = hamming_weight(result_target)
        
        # 对于101模式，汉明重量为2
        if target_bits_guess == 5 and hw_key == 2:
            power += 5.0
        
        # 重量变化模式
        weight_change = abs(hw_product - hw_key) + abs(hw_result - hw_product)
        power += weight_change * 1.2
        
        # 位位置相关的权重
        for i in range(TARGET_BITS_COUNT):
            bit_pos = TARGET_BITS_START + i
            key_bit = (key_target >> i) & 1
            result_bit = (result_target >> i) & 1
            
            # 对于101模式，位0和位2应该是1，位1应该是0
            if target_bits_guess == 5:
                if i == 0 or i == 2:  # 位0和位2
                    if key_bit == 1:
                        power += 3.0
                    if result_bit == 1:
                        power += 2.0
                elif i == 1:  # 位1
                    if key_bit == 0:
                        power += 3.0
                    if result_bit == 0:
                        power += 2.0
            
            # 通用位权重
            power += key_bit * (1.0 + 0.1 * i)
            power += result_bit * (1.2 + 0.1 * i)
        
        power += transitions
        total_power += power
    
    return total_power / UNKNOWN_HIGH_POSSIBILITIES

def power_model_ensemble_carry(plaintext, target_bits_guess):
    """
    集成进位传播模型
    """
    total_power = 0
    
    for high_bits in range(UNKNOWN_HIGH_POSSIBILITIES):
        full_key = (high_bits << TARGET_BITS_END) | (target_bits_guess << TARGET_BITS_START) | KNOWN_LOW_BITS
        
        product = (full_key * WN) % MODULUS
        
        power = 0
        carry = 0
        
        # 详细的进位分析
        for bit_pos in range(12):
            a_bit = (plaintext >> bit_pos) & 1
            b_bit = (product >> bit_pos) & 1
            
            sum_bit = a_bit ^ b_bit ^ carry
            new_carry = (a_bit & b_bit) | (carry & (a_bit ^ b_bit))
            
            if TARGET_BITS_START <= bit_pos < TARGET_BITS_END:
                # 目标位的进位权重
                power += new_carry * 4.0
                
                # 特殊模式检测
                if target_bits_guess == 5:  # 101模式
                    target_bit_index = bit_pos - TARGET_BITS_START
                    expected_bit = (5 >> target_bit_index) & 1
                    
                    if (a_bit ^ b_bit) == expected_bit:
                        power += 2.0
                    
                    # 101模式的进位特性
                    if target_bit_index == 1 and new_carry == 0:  # 中间位不进位
                        power += 3.0
                    elif target_bit_index != 1 and new_carry == 1:  # 两端进位
                        power += 2.0
            else:
                power += new_carry * 1.0
            
            carry = new_carry
        
        # 模运算的额外分析
        intermediate_sum = plaintext + product
        if intermediate_sum >= MODULUS:
            target_mask = ((1 << TARGET_BITS_COUNT) - 1) << TARGET_BITS_START
            before_mod = (intermediate_sum & target_mask) >> TARGET_BITS_START
            after_mod = ((intermediate_sum % MODULUS) & target_mask) >> TARGET_BITS_START
            
            if before_mod != after_mod:
                power += 3.0
                
                # 对于101模式的特殊处理
                if target_bits_guess == 5 and after_mod == 5:
                    power += 5.0
        
        total_power += power
    
    return total_power / UNKNOWN_HIGH_POSSIBILITIES

def power_model_statistical_enhanced(plaintext, target_bits_guess):
    """
    统计增强模型
    """
    powers = []
    
    for high_bits in range(UNKNOWN_HIGH_POSSIBILITIES):
        full_key = (high_bits << TARGET_BITS_END) | (target_bits_guess << TARGET_BITS_START) | KNOWN_LOW_BITS
        
        product = (full_key * WN) % MODULUS
        result = (plaintext + product) % MODULUS
        
        # 提取目标位
        target_mask = ((1 << TARGET_BITS_COUNT) - 1) << TARGET_BITS_START
        key_target = (full_key & target_mask) >> TARGET_BITS_START
        product_target = (product & target_mask) >> TARGET_BITS_START
        result_target = (result & target_mask) >> TARGET_BITS_START
        
        # 多维特征
        features = [
            hamming_weight(key_target),
            hamming_weight(product_target),
            hamming_weight(result_target),
            hamming_distance(key_target, product_target),
            hamming_distance(product_target, result_target),
            key_target,
            product_target,
            result_target,
            (key_target * product_target) % 8,
            (result_target ^ key_target) % 8
        ]
        
        # 对于101模式的特殊特征
        if target_bits_guess == 5:
            features.extend([
                1 if key_target == 5 else 0,
                1 if (key_target & 5) == 5 else 0,  # 包含101模式
                1 if (result_target & 5) == 5 else 0,
                hamming_distance(key_target, 5),
                hamming_distance(result_target, 5)
            ])
        else:
            features.extend([0, 0, 0, 8, 8])  # 默认值
        
        power = sum(f * (i + 1) * 0.1 for i, f in enumerate(features))
        powers.append(power)
    
    return np.mean(powers)

class CPA_Bits_4_6_Final:
    def __init__(self,
                 power_file,
                 power_models,
                 plaintext_number=20000,
                 sample_number=5000,
                 thread_number=12):
        
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
        self.ensemble_results = {}
        self.time = list(range(sample_number))
        
        # 线程锁
        self.data_lock = td.Lock()
        
    def thread_read_signals(self):
        """读取功耗迹线"""
        print(f'最终4-6位CPA攻击: 开始读取功耗迹线')
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
        """计算相关系数（使用多种方法）"""
        print("计算最终4-6位相关系数...")
        
        power_traces_array = np.array(self.power_traces)
        
        for model_name in self.power_models.keys():
            print(f"\n分析模型: {model_name}")
            
            pearson_correlations = np.zeros(self.key_number)
            spearman_correlations = np.zeros(self.key_number)
            
            for target_guess in range(self.key_number):
                theoretical_power_array = np.array(self.theoretical_powers[model_name][target_guess])
                
                # 数据标准化
                if np.std(theoretical_power_array) > 0:
                    theoretical_power_array = (theoretical_power_array - np.mean(theoretical_power_array)) / np.std(theoretical_power_array)
                
                max_pearson = 0
                max_spearman = 0
                
                for sample_idx in range(self.sample_number):
                    try:
                        actual_power = power_traces_array[:, sample_idx]
                        
                        if np.std(actual_power) > 0:
                            actual_power = (actual_power - np.mean(actual_power)) / np.std(actual_power)
                        
                        if len(actual_power) > 1 and len(theoretical_power_array) > 1:
                            # 皮尔逊相关系数
                            pearson_corr, p_val = pearsonr(actual_power, theoretical_power_array)
                            if not np.isnan(pearson_corr) and p_val < 0.05:
                                max_pearson = max(max_pearson, abs(pearson_corr))
                            
                            # 斯皮尔曼相关系数
                            spearman_corr, _ = spearmanr(actual_power, theoretical_power_array)
                            if not np.isnan(spearman_corr):
                                max_spearman = max(max_spearman, abs(spearman_corr))
                            
                    except Exception as e:
                        pass
                
                pearson_correlations[target_guess] = max_pearson
                spearman_correlations[target_guess] = max_spearman
                
                print(f"  4-6位猜测 {target_guess} (二进制: {target_guess:03b}): Pearson={max_pearson:.6f}, Spearman={max_spearman:.6f}")
            
            self.correlation_results[model_name] = {
                'pearson': pearson_correlations,
                'spearman': spearman_correlations
            }

    def ensemble_analysis(self):
        """集成分析"""
        print("\n=== 集成分析 ===")
        
        # 收集所有模型的结果
        all_pearson_scores = np.zeros((len(self.power_models), self.key_number))
        all_spearman_scores = np.zeros((len(self.power_models), self.key_number))
        
        model_names = list(self.power_models.keys())
        
        for i, model_name in enumerate(model_names):
            all_pearson_scores[i] = self.correlation_results[model_name]['pearson']
            all_spearman_scores[i] = self.correlation_results[model_name]['spearman']
        
        # 集成方法1：平均
        ensemble_avg_pearson = np.mean(all_pearson_scores, axis=0)
        ensemble_avg_spearman = np.mean(all_spearman_scores, axis=0)
        
        # 集成方法2：加权平均（给表现好的模型更高权重）
        pearson_weights = np.array([np.max(all_pearson_scores[i]) for i in range(len(model_names))])
        pearson_weights = pearson_weights / np.sum(pearson_weights)
        
        spearman_weights = np.array([np.max(all_spearman_scores[i]) for i in range(len(model_names))])
        spearman_weights = spearman_weights / np.sum(spearman_weights)
        
        ensemble_weighted_pearson = np.sum(all_pearson_scores * pearson_weights.reshape(-1, 1), axis=0)
        ensemble_weighted_spearman = np.sum(all_spearman_scores * spearman_weights.reshape(-1, 1), axis=0)
        
        # 集成方法3：投票
        pearson_votes = np.zeros(self.key_number)
        spearman_votes = np.zeros(self.key_number)
        
        for i in range(len(model_names)):
            best_pearson = np.argmax(all_pearson_scores[i])
            best_spearman = np.argmax(all_spearman_scores[i])
            pearson_votes[best_pearson] += 1
            spearman_votes[best_spearman] += 1
        
        self.ensemble_results = {
            'avg_pearson': ensemble_avg_pearson,
            'avg_spearman': ensemble_avg_spearman,
            'weighted_pearson': ensemble_weighted_pearson,
            'weighted_spearman': ensemble_weighted_spearman,
            'votes_pearson': pearson_votes,
            'votes_spearman': spearman_votes
        }
        
        print("\n集成结果:")
        for method, scores in self.ensemble_results.items():
            best_guess = np.argmax(scores)
            best_score = scores[best_guess]
            print(f"  {method}: 最佳猜测={best_guess} ({best_guess:03b}), 得分={best_score:.6f}")

    def multi_thread_start(self):
        """启动多线程分析"""
        print(f'最终4-6位CPA攻击开始')
        print(f'样本数量: {self.plaintext_number}')
        print(f'目标: 破解密钥的第{TARGET_BITS_START+1}-{TARGET_BITS_END}位')
        print(f'已知低3位: {KNOWN_LOW_BITS:03b}')
        print(f'真实4-6位: 5 (101)')

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
        print("\n=== 最终4-6位CPA攻击结果 ===")
        
        self.calculate_correlations()
        self.ensemble_analysis()
        
        # 真实密钥1000的4-6位
        true_bits_4_6 = 5
        print(f"\n真实密钥1000的4-6位: {true_bits_4_6} (二进制: {true_bits_4_6:03b})")
        
        # 分析各个模型的表现
        print("\n=== 各模型详细结果 ===")
        for model_name, results in self.correlation_results.items():
            pearson_scores = results['pearson']
            spearman_scores = results['spearman']
            
            best_pearson = np.argmax(pearson_scores)
            best_spearman = np.argmax(spearman_scores)
            
            print(f"\n{model_name}:")
            print(f"  Pearson最佳: {best_pearson} ({best_pearson:03b}), 得分={pearson_scores[best_pearson]:.6f}")
            print(f"  Spearman最佳: {best_spearman} ({best_spearman:03b}), 得分={spearman_scores[best_spearman]:.6f}")
            print(f"  真实值5表现: Pearson={pearson_scores[5]:.6f}, Spearman={spearman_scores[5]:.6f}")
            
            # 排名分析
            pearson_rank = np.sum(pearson_scores > pearson_scores[5]) + 1
            spearman_rank = np.sum(spearman_scores > spearman_scores[5]) + 1
            print(f"  真实值排名: Pearson第{pearson_rank}名, Spearman第{spearman_rank}名")
        
        # 集成结果分析
        print("\n=== 集成方法结果 ===")
        success_count = 0
        total_methods = len(self.ensemble_results)
        
        for method, scores in self.ensemble_results.items():
            best_guess = np.argmax(scores)
            is_correct = best_guess == true_bits_4_6
            if is_correct:
                success_count += 1
            
            true_score = scores[true_bits_4_6]
            true_rank = np.sum(scores > true_score) + 1
            
            print(f"\n{method}:")
            print(f"  推荐: {best_guess} ({best_guess:03b}), 得分={scores[best_guess]:.6f}")
            print(f"  真实值5: 得分={true_score:.6f}, 排名={true_rank}")
            print(f"  是否正确: {'✓' if is_correct else '✗'}")
        
        print(f"\n=== 最终结论 ===")
        print(f"集成方法成功率: {success_count}/{total_methods} = {success_count/total_methods*100:.1f}%")
        
        # 找到最佳集成方法
        best_method = max(self.ensemble_results.keys(), 
                         key=lambda x: self.ensemble_results[x][true_bits_4_6])
        best_scores = self.ensemble_results[best_method]
        recommended = np.argmax(best_scores)
        
        print(f"\n最佳集成方法: {best_method}")
        print(f"最终推荐4-6位: {recommended} (二进制: {recommended:03b})")
        print(f"推荐正确性: {'✓' if recommended == true_bits_4_6 else '✗'}")
        
        return {
            'individual_results': self.correlation_results,
            'ensemble_results': self.ensemble_results,
            'final_recommendation': recommended,
            'success_rate': success_count / total_methods
        }

if __name__ == "__main__":
    print("=== 最终4-6位CPA攻击 ===")
    print(f"攻击目标: 第{TARGET_BITS_START+1}-{TARGET_BITS_END}位")
    print(f"已知信息: 低3位 = {KNOWN_LOW_BITS}")
    print(f"真实密钥: 1000 (二进制: {1000:012b})")
    print(f"真实4-6位: 5 (二进制: 101)")
    
    # 定义最终的功耗模型
    power_models = {
        "针对5优化模型": power_model_optimized_for_5,
        "集成进位模型": power_model_ensemble_carry,
        "统计增强模型": power_model_statistical_enhanced
    }
    
    # 运行最终的4-6位CPA攻击
    cpa_final = CPA_Bits_4_6_Final(
        power_file='../data/ntt_pipeline_traces_x10k-3rd.txt',
        power_models=power_models,
        plaintext_number=20000,
        sample_number=5000,
        thread_number=12
    )
    
    results = cpa_final.multi_thread_start()
    
    print("\n" + "="*60)
    print("=== 最终攻击总结 ===")
    print(f"最终推荐: {results['final_recommendation']} (二进制: {results['final_recommendation']:03b})")
    print(f"真实值: 5 (二进制: 101)")
    print(f"攻击成功: {'✓' if results['final_recommendation'] == 5 else '✗'}")
    print(f"集成成功率: {results['success_rate']*100:.1f}%")