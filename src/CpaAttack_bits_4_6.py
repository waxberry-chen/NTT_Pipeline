#!/usr/bin/env python3
"""
使用位转换模型破解密钥的{TARGET_BITS_START+1}-{TARGET_BITS_END}位
基于成功的位转换模型，扩展攻击范围到密钥的第{TARGET_BITS_START+1}-{TARGET_BITS_END}位
"""

import os
import queue
import threading as td
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

WN = 1729
MODULUS = 3329

# 针对{TARGET_BITS_START+1}-{TARGET_BITS_END}位的攻击参数
TARGET_BITS_START = 8  # 从第3位开始（0-indexed，即第4位）
TARGET_BITS_END = 11    # 到第5位结束（0-indexed，即第6位）
TARGET_BITS_COUNT = TARGET_BITS_END - TARGET_BITS_START  # 3位
TARGET_POSSIBILITIES = 1 << TARGET_BITS_COUNT  # 2^3 = 8种可能

# 已知的低3位（从之前的攻击结果）
KNOWN_LOW_BITS = 232  # 密钥1000的低3位是0

# 未知的高位数量
UNKNOWN_HIGH_BITS = 12 - TARGET_BITS_END  # 12 - 6 = 6位
UNKNOWN_HIGH_POSSIBILITIES = 1 << UNKNOWN_HIGH_BITS  # 2^6 = 64种可能

def hamming_distance(x, y):
    """计算汉明距离"""
    return bin(x ^ y).count('1')

def hamming_weight(x):
    """计算汉明重量"""
    return bin(x).count('1')

def power_model_bit_transitions_4_6(plaintext, target_bits_guess):
    """
    针对{TARGET_BITS_START+1}-{TARGET_BITS_END}位的位转换功耗模型
    
    Args:
        plaintext (int): 输入明文
        target_bits_guess (int): 对{TARGET_BITS_START+1}-{TARGET_BITS_END}位的猜测 (0-7)
    
    Returns:
        float: 理论功耗值
    """
    total_transitions = 0
    
    # 遍历所有可能的高位组合
    for high_bits in range(UNKNOWN_HIGH_POSSIBILITIES):
        # 重构完整的12位密钥
        # 结构: [高6位][目标3位({TARGET_BITS_START+1}-{TARGET_BITS_END})][已知低3位]
        full_key = (high_bits << TARGET_BITS_END) | (target_bits_guess << TARGET_BITS_START) | KNOWN_LOW_BITS
        
        # NTT运算的各个步骤
        step1 = full_key * WN                    # 乘法
        step2 = step1 % MODULUS                  # 第一次模运算
        step3 = plaintext + step2                # 加法
        step4 = step3 % MODULUS                  # 最终模运算
        
        # 计算每一步的位转换
        transitions = 0
        
        # 1. 密钥加载到乘法器的位转换
        transitions += hamming_distance(0, full_key)  # 从0状态加载密钥
        
        # 2. 乘法运算的位转换（关注中间结果的位变化）
        transitions += hamming_distance(full_key, step1 & 0xFFF)  # 限制在12位内
        
        # 3. 第一次模运算的位转换
        transitions += hamming_distance(step1 & 0xFFF, step2)
        
        # 4. 加法器的位转换
        transitions += hamming_distance(step2, step3 & 0xFFF)
        
        # 5. 最终模运算的位转换
        transitions += hamming_distance(step3 & 0xFFF, step4)
        
        # 6. 特别关注目标位区域的转换
        target_mask = ((1 << TARGET_BITS_COUNT) - 1) << TARGET_BITS_START
        
        # 提取各步骤中目标位的值
        target_step1 = (step1 & target_mask) >> TARGET_BITS_START
        target_step2 = (step2 & target_mask) >> TARGET_BITS_START
        target_step3 = (step3 & target_mask) >> TARGET_BITS_START
        target_step4 = (step4 & target_mask) >> TARGET_BITS_START
        
        # 目标位区域的额外转换权重
        target_transitions = (
            hamming_distance(target_bits_guess, target_step1) +
            hamming_distance(target_step1, target_step2) +
            hamming_distance(target_step2, target_step3) +
            hamming_distance(target_step3, target_step4)
        )
        
        # 加权组合：总体转换 + 目标位转换的额外权重
        total_transitions += transitions + 2.0 * target_transitions
    
    return total_transitions / UNKNOWN_HIGH_POSSIBILITIES

def power_model_target_bits_focus(plaintext, target_bits_guess):
    """
    专注于目标位的功耗模型
    重点分析{TARGET_BITS_START+1}-{TARGET_BITS_END}位在NTT运算中的行为
    """
    total_power = 0
    
    for high_bits in range(UNKNOWN_HIGH_POSSIBILITIES):
        # 重构完整密钥
        full_key = (high_bits << TARGET_BITS_END) | (target_bits_guess << TARGET_BITS_START) | KNOWN_LOW_BITS
        
        # NTT运算
        product = (full_key * WN) % MODULUS
        result = (plaintext + product) % MODULUS
        
        # 提取目标位区域
        target_mask = ((1 << TARGET_BITS_COUNT) - 1) << TARGET_BITS_START
        
        # 分析目标位在各个运算中的贡献
        key_target_bits = (full_key & target_mask) >> TARGET_BITS_START
        product_target_bits = (product & target_mask) >> TARGET_BITS_START
        result_target_bits = (result & target_mask) >> TARGET_BITS_START
        
        # 目标位的功耗贡献
        power = (
            hamming_weight(key_target_bits) * 1.0 +      # 密钥位的权重
            hamming_weight(product_target_bits) * 1.5 +   # 乘法结果的权重
            hamming_weight(result_target_bits) * 2.0 +    # 最终结果的权重
            hamming_distance(key_target_bits, product_target_bits) * 1.2 +  # 乘法转换
            hamming_distance(product_target_bits, result_target_bits) * 1.3  # 加法转换
        )
        
        total_power += power
    
    return total_power / UNKNOWN_HIGH_POSSIBILITIES

def power_model_carry_propagation(plaintext, target_bits_guess):
    """
    基于进位传播的功耗模型
    关注{TARGET_BITS_START+1}-{TARGET_BITS_END}位在加法运算中的进位行为
    """
    total_carry_power = 0
    
    for high_bits in range(UNKNOWN_HIGH_POSSIBILITIES):
        full_key = (high_bits << TARGET_BITS_END) | (target_bits_guess << TARGET_BITS_START) | KNOWN_LOW_BITS
        
        product = (full_key * WN) % MODULUS
        
        # 分析加法运算中的进位传播
        # 逐位计算加法，关注进位链
        carry = 0
        carry_count = 0
        
        for bit_pos in range(12):  # 12位加法
            a_bit = (plaintext >> bit_pos) & 1
            b_bit = (product >> bit_pos) & 1
            
            sum_bit = a_bit ^ b_bit ^ carry
            new_carry = (a_bit & b_bit) | (carry & (a_bit ^ b_bit))
            
            # 如果当前位在目标范围内，增加权重
            if TARGET_BITS_START <= bit_pos < TARGET_BITS_END:
                carry_count += new_carry * 3  # 目标位的进位权重更高
            else:
                carry_count += new_carry
            
            carry = new_carry
        
        # 最终模运算的额外功耗
        intermediate_sum = plaintext + product
        if intermediate_sum >= MODULUS:
            carry_count += 2  # 模运算减法的额外功耗
        
        total_carry_power += carry_count
    
    return total_carry_power / UNKNOWN_HIGH_POSSIBILITIES

class CPA_Bits_4_6:
    def __init__(self,
                 power_file,
                 power_models,
                 plaintext_number=4096,
                 sample_number=5000,
                 thread_number=10):
        
        self.power_file = power_file
        self.power_models = power_models
        self.key_number = TARGET_POSSIBILITIES  # 8种可能的{TARGET_BITS_START+1}-{TARGET_BITS_END}位组合
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
        print(f'{TARGET_BITS_START+1}-{TARGET_BITS_END}位CPA攻击: 开始读取功耗迹线')
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
        """计算相关系数"""
        print("计算{TARGET_BITS_START+1}-{TARGET_BITS_END}位的相关系数...")
        
        power_traces_array = np.array(self.power_traces)
        
        for model_name in self.power_models.keys():
            print(f"\n分析模型: {model_name}")
            
            correlation_matrix = np.zeros((self.key_number, self.sample_number))
            max_correlations = np.zeros(self.key_number)
            
            for target_guess in range(self.key_number):
                theoretical_power_array = np.array(self.theoretical_powers[model_name][target_guess])
                
                for sample_idx in range(self.sample_number):
                    try:
                        actual_power = power_traces_array[:, sample_idx]
                        
                        if len(actual_power) > 1 and len(theoretical_power_array) > 1:
                            correlation, _ = pearsonr(actual_power, theoretical_power_array)
                            if not np.isnan(correlation):
                                correlation_matrix[target_guess, sample_idx] = abs(correlation)
                            else:
                                correlation_matrix[target_guess, sample_idx] = 0
                        else:
                            correlation_matrix[target_guess, sample_idx] = 0
                            
                    except Exception as e:
                        correlation_matrix[target_guess, sample_idx] = 0
                
                max_correlations[target_guess] = np.max(correlation_matrix[target_guess, :])
                print(f"  {TARGET_BITS_START+1}-{TARGET_BITS_END}位猜测 {target_guess} (二进制: {target_guess:03b}): 最大相关系数 = {max_correlations[target_guess]:.6f}")
            
            self.correlation_results[model_name] = {
                'matrix': correlation_matrix,
                'max_correlations': max_correlations
            }

    def multi_thread_start(self):
        """启动多线程分析"""
        print(f'{TARGET_BITS_START+1}-{TARGET_BITS_END}位CPA攻击开始')
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
        print("\n=== {TARGET_BITS_START+1}-{TARGET_BITS_END}位CPA攻击结果 ===")
        
        self.calculate_correlations()
        
        best_results = {}
        
        # 真实密钥1000的{TARGET_BITS_START+1}-{TARGET_BITS_END}位
        true_key = 1000
        true_bits_4_6 = (true_key >> TARGET_BITS_START) & ((1 << TARGET_BITS_COUNT) - 1)
        print(f"\n真实密钥1000的{TARGET_BITS_START+1}-{TARGET_BITS_END}位: {true_bits_4_6} (二进制: {true_bits_4_6:03b})")
        
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
            
            # 显示真实{TARGET_BITS_START+1}-{TARGET_BITS_END}位的表现
            true_corr = max_correlations[true_bits_4_6]
            true_rank = np.sum(max_correlations > true_corr) + 1
            print(f"  真实{TARGET_BITS_START+1}-{TARGET_BITS_END}位 {true_bits_4_6}: 相关系数={true_corr:.6f}, 排名={true_rank}")
            
            # 显示所有候选的排序
            sorted_indices = np.argsort(max_correlations)[::-1]
            print(f"  排序结果:")
            for i, idx in enumerate(sorted_indices[:5]):  # 显示前5名
                marker = " ← 真实" if idx == true_bits_4_6 else ""
                print(f"    {i+1}. 猜测{idx} ({idx:03b}): {max_correlations[idx]:.6f}{marker}")
        
        # 找到总体最佳模型
        best_model = max(best_results.keys(), 
                        key=lambda x: best_results[x]['best_correlation'])
        
        print(f"\n=== 最终推荐 ===")
        print(f"最佳模型: {best_model}")
        print(f"推荐{TARGET_BITS_START+1}-{TARGET_BITS_END}位: {best_results[best_model]['best_guess']} (二进制: {best_results[best_model]['best_guess']:03b})")
        print(f"置信度: {best_results[best_model]['best_correlation']:.6f}")
        
        # 重构完整密钥猜测
        best_4_6_bits = best_results[best_model]['best_guess']
        reconstructed_key_low_9_bits = (best_4_6_bits << TARGET_BITS_START) | KNOWN_LOW_BITS
        print(f"\n重构的低9位密钥: {reconstructed_key_low_9_bits} (二进制: {reconstructed_key_low_9_bits:09b})")
        print(f"真实低9位: {true_key & 0x1FF} (二进制: {true_key & 0x1FF:09b})")
        
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
                title = f"{model_name}\n{TARGET_BITS_START+1}-{TARGET_BITS_END}位: {guess} ({guess:03b})"
                if guess == true_bits_4_6:
                    title += " (真实)"
                title += f"\nMax: {max_correlations[guess]:.4f}"
                ax.set_title(title, fontsize=8)
                ax.set_xlabel("Sample Point")
                ax.set_ylabel("Correlation")
                ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('cpa_bits_4_6_results.png', dpi=300, bbox_inches='tight')
        print("{TARGET_BITS_START+1}-{TARGET_BITS_END}位攻击结果图已保存为 cpa_bits_4_6_results.png")

if __name__ == "__main__":
    print("=== 使用位转换模型破解密钥{TARGET_BITS_START+1}-{TARGET_BITS_END}位 ===")
    print(f"攻击目标: 第{TARGET_BITS_START+1}-{TARGET_BITS_END}位")
    print(f"已知信息: 低3位 = {KNOWN_LOW_BITS}")
    print(f"真实密钥: 1000 (二进制: {1000:012b})")
    print(f"真实{TARGET_BITS_START+1}-{TARGET_BITS_END}位: {(1000 >> TARGET_BITS_START) & 0x7} (二进制: {(1000 >> TARGET_BITS_START) & 0x7:03b})")
    
    # 定义针对{TARGET_BITS_START+1}-{TARGET_BITS_END}位的功耗模型
    power_models = {
        "位转换模型({TARGET_BITS_START+1}-{TARGET_BITS_END}位)": power_model_bit_transitions_4_6,
        "目标位专注模型": power_model_target_bits_focus,
        "进位传播模型": power_model_carry_propagation
    }
    
    # 运行{TARGET_BITS_START+1}-{TARGET_BITS_END}位CPA攻击
    cpa_4_6 = CPA_Bits_4_6(
        power_file='../data/ntt_pipeline_traces_x10k-3rd.txt',
        power_models=power_models,
        plaintext_number=10000,
        sample_number=5000,
        thread_number=10
    )
    
    results = cpa_4_6.multi_thread_start()
    
    print("\n" + "="*60)
    print("=== {TARGET_BITS_START+1}-{TARGET_BITS_END}位攻击总结 ===")
    for model_name, result in results.items():
        true_bits_4_6 = (1000 >> TARGET_BITS_START) & 0x7
        true_corr = result['all_correlations'][true_bits_4_6]
        best_guess = result['best_guess']
        best_corr = result['best_correlation']
        
        print(f"\n{model_name}:")
        print(f"  推荐{TARGET_BITS_START+1}-{TARGET_BITS_END}位: {best_guess} ({best_guess:03b}), 相关系数: {best_corr:.6f}")
        print(f"  真实{TARGET_BITS_START+1}-{TARGET_BITS_END}位表现: {true_bits_4_6} ({true_bits_4_6:03b}), 相关系数: {true_corr:.6f}")
        print(f"  是否正确: {'✓' if best_guess == true_bits_4_6 else '✗'}")