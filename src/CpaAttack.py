import os
import queue
import threading as td
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

WN = 1729
MODULUS = 3329
# Unknown bit number
NUM_UNKNOWN_BITS = 9  # 12-bit key - 3-bit guess = 9 unknown bits
UNKNOWN_POSSIBILITIES = 1 << NUM_UNKNOWN_BITS # 2^9 = 512

def hamming_weight(x):
    """计算汉明重量（二进制中1的个数）"""
    return bin(x).count('1')

def power_model_hw(plaintext, key_guess):
    """
    基于汉明重量的功耗模型
    
    Args:
        plaintext (int): 输入的明文 a
        key_guess (int): 对密钥低3位的猜测 (0-7)
    
    Returns:
        float: 理论功耗值（基于汉明重量）
    """
    # 计算所有可能高位的平均汉明重量
    total_hw = 0
    
    for b_high in range(UNKNOWN_POSSIBILITIES):
        # 重构完整的12位密钥
        b_full_guess = (b_high << 3) | key_guess
        # 计算NTT运算结果
        product_guess = (b_full_guess * WN) % MODULUS
        result = (plaintext + product_guess) % MODULUS
        # 累加汉明重量
        total_hw += hamming_weight(result)
    
    # 返回平均汉明重量
    return total_hw / UNKNOWN_POSSIBILITIES

def power_model_transition(plaintext, key_guess):
    """
    基于状态转换的功耗模型
    
    Args:
        plaintext (int): 输入的明文 a
        key_guess (int): 对密钥低3位的猜测 (0-7)
    
    Returns:
        float: 理论功耗值（基于状态转换）
    """
    # 计算所有可能高位的平均转换次数
    total_transitions = 0
    
    for b_high in range(UNKNOWN_POSSIBILITIES):
        # 重构完整的12位密钥
        b_full_guess = (b_high << 3) | key_guess
        # 计算NTT运算结果
        product_guess = (b_full_guess * WN) % MODULUS
        result = (plaintext + product_guess) % MODULUS
        
        # 计算相邻位之间的转换次数
        transitions = 0
        prev_bit = result & 1
        for i in range(1, 12):  # 假设12位数据
            curr_bit = (result >> i) & 1
            if curr_bit != prev_bit:
                transitions += 1
            prev_bit = curr_bit
        
        total_transitions += transitions
    
    return total_transitions / UNKNOWN_POSSIBILITIES

class CPA:
    def __init__(self,
                 power_file,
                 power_model,
                 key_number,
                 plaintext_number=4096,
                 sample_number=5000,
                 thread_number=10):
        # 基本参数
        self.power_file = power_file
        self.power_model = power_model
        self.key_number = key_number
        self.plaintext_number = plaintext_number
        self.sample_number = sample_number
        self.thread_number = thread_number
        self.threads = {}
        self.q = queue.Queue()

        # CPA数据存储
        self.power_traces = []  # 存储所有功耗迹线
        self.theoretical_powers = [[] for _ in range(key_number)]  # 每个密钥猜测的理论功耗
        self.plaintexts = []  # 存储所有明文
        
        # 结果存储
        self.correlation_matrix = np.zeros((key_number, sample_number))  # 相关系数矩阵
        self.max_correlations = np.zeros(key_number)  # 每个密钥的最大相关系数
        self.time = list(range(sample_number))

        # 线程锁
        self.data_lock = td.Lock()
        
    def thread_read_signals(self):
        """读取功耗迹线的线程"""
        print(f'CPA: thread power trace read start')
        number = 0
        with open(self.power_file, 'r') as pf:
            for line in pf:
                if (number >= self.plaintext_number) or (not line.strip()):
                    break
                try:
                    plaintext_str, power_trace_str = line.split(':', 1)
                    plaintext = int(plaintext_str, 10)  # 明文是十进制格式
                    power_trace = np.array(power_trace_str.strip().split()).astype(np.float64)
                    
                    # 调整功耗迹线长度
                    if len(power_trace) < self.sample_number:
                        power_trace = np.pad(power_trace, (0, self.sample_number - len(power_trace)))
                    elif len(power_trace) > self.sample_number:
                        power_trace = power_trace[:self.sample_number]
                    
                    # 存储数据
                    signal_data = {'plaintext': plaintext, 'power_trace': power_trace}
                    self.q.put(signal_data)
                    
                    print(f'CPA: read add {plaintext:04X}')
                    number += 1
                except Exception as e:
                    print(f"CPA: Error parsing line: {line.strip()} - {str(e)}")
        
        # 发送退出信号
        for _ in range(self.thread_number):
            self.q.put({'plaintext': 'EXIT_SIGNAL', 'power_trace': None})
        print(f'CPA: thread read end, added {number} traces')

    def thread_signals_analyze(self, thread_id):
        """分析功耗迹线的线程"""
        print(f'CPA: thread analyze {thread_id} start')
        processed_count = 0
        local_power_traces = []
        local_plaintexts = []
        local_theoretical_powers = [[] for _ in range(self.key_number)]
        
        while True:
            try:
                signal = self.q.get(timeout=10)
                if signal['plaintext'] == 'EXIT_SIGNAL':
                    print(f'CPA: thread {thread_id} received exit signal')
                    self.q.task_done()
                    break
                
                plaintext = signal['plaintext']
                power_trace = signal['power_trace']
                
                # 存储数据
                local_power_traces.append(power_trace)
                local_plaintexts.append(plaintext)
                
                # 计算每个密钥猜测的理论功耗
                for key in range(self.key_number):
                    theoretical_power = self.power_model(plaintext, key)
                    local_theoretical_powers[key].append(theoretical_power)
                
                print(f'CPA: thread {thread_id} processed data {plaintext:04X}')
                processed_count += 1
                self.q.task_done()
                
            except queue.Empty:
                print(f"CPA: Thread {thread_id} queue empty, exiting")
                break
            except Exception as e:
                print(f"CPA: Thread {thread_id} general error: {str(e)}")
        
        # 将本地数据合并到全局数据
        with self.data_lock:
            self.power_traces.extend(local_power_traces)
            self.plaintexts.extend(local_plaintexts)
            for key in range(self.key_number):
                self.theoretical_powers[key].extend(local_theoretical_powers[key])
        
        print(f'CPA: thread analyze {thread_id} end, processed {processed_count} traces')

    def calculate_correlations(self):
        """计算相关系数"""
        print("CPA: Calculating correlations...")
        
        # 转换为numpy数组
        power_traces_array = np.array(self.power_traces)  # shape: (num_traces, sample_number)
        
        for key in range(self.key_number):
            theoretical_power_array = np.array(self.theoretical_powers[key])  # shape: (num_traces,)
            
            # 计算每个采样点的相关系数
            for sample_idx in range(self.sample_number):
                try:
                    # 获取所有迹线在该采样点的功耗值
                    actual_power = power_traces_array[:, sample_idx]
                    
                    # 计算皮尔逊相关系数
                    if len(actual_power) > 1 and len(theoretical_power_array) > 1:
                        correlation, _ = pearsonr(actual_power, theoretical_power_array)
                        if not np.isnan(correlation):
                            self.correlation_matrix[key, sample_idx] = abs(correlation)
                        else:
                            self.correlation_matrix[key, sample_idx] = 0
                    else:
                        self.correlation_matrix[key, sample_idx] = 0
                        
                except Exception as e:
                    print(f"CPA: Error calculating correlation for key {key}, sample {sample_idx}: {str(e)}")
                    self.correlation_matrix[key, sample_idx] = 0
            
            # 记录每个密钥的最大相关系数
            self.max_correlations[key] = np.max(self.correlation_matrix[key, :])
            print(f"CPA: Key {key}: max correlation = {self.max_correlations[key]:.6f}")

    def multi_thread_start(self):
        """启动多线程CPA分析"""
        print(f'CPA: start main process: {os.getpid()}')

        # 启动读取线程
        reader = td.Thread(target=self.thread_read_signals)
        reader.start()
        self.threads['reader'] = reader

        # 启动分析线程
        analyzers = []
        for thread_id in range(self.thread_number):
            analyzer = td.Thread(
                target=self.thread_signals_analyze,
                args=(thread_id,)
            )
            analyzer.daemon = True
            analyzer.start()
            analyzers.append(analyzer)
            self.threads[thread_id] = analyzer

        # 等待所有线程完成
        reader.join()
        print("CPA: Reader thread completed")
        self.q.join()
        print("CPA: All queue tasks completed")

        for analyzer in analyzers:
            analyzer.join(timeout=5)

        print("\nCPA: 所有线程已完成")

        # 分析结果
        best_key, best_correlation = self.result_analyze()
        print(f'CPA: end main process: {os.getpid()}')
        return best_key, best_correlation

    def result_analyze(self):
        """分析CPA结果"""
        print("\nCPA: 开始相关功耗分析...")
        
        # 计算相关系数
        self.calculate_correlations()
        
        # 找到最佳密钥猜测
        best_key = np.argmax(self.max_correlations)
        best_correlation = self.max_correlations[best_key]
        
        print("\n=== CPA分析结果 ===")
        print(f"最佳密钥猜测: {best_key} (0x{best_key:X})")
        print(f"最大相关系数: {best_correlation:.6f}")
        
        print("\n所有密钥的最大相关系数:")
        for key in range(self.key_number):
            print(f"Key {key} (0x{key:X}): {self.max_correlations[key]:.6f}")
        
        # 绘制结果
        self.plot_results()
        
        return best_key, best_correlation

    def plot_results(self):
        """绘制CPA结果"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for key in range(self.key_number):
            axes[key].plot(self.time, self.correlation_matrix[key, :])
            axes[key].set_title(f"Key {key} (0x{key:X})\nMax: {self.max_correlations[key]:.4f}")
            axes[key].set_xlabel("Sample Point")
            axes[key].set_ylabel("Correlation")
            axes[key].grid(True)
        
        plt.tight_layout()
        plt.savefig('cpa_results.png', dpi=300, bbox_inches='tight')
        print("CPA: 结果图已保存为 cpa_results.png")
        # plt.show()

if __name__ == "__main__":
    print("=== 开始CPA攻击 ===")
    
    # 使用汉明重量模型
    print("\n使用汉明重量功耗模型...")
    cpa_hw = CPA(
        power_file='../data/ntt_pipeline_traces_x10k-3rd.txt',
        power_model=power_model_hw,
        key_number=8,
        plaintext_number=10000,
        sample_number=5000,
        thread_number=10
    )
    best_key_hw, best_corr_hw = cpa_hw.multi_thread_start()
    
    print("\n" + "="*50)
    
    # 使用状态转换模型
    print("\n使用状态转换功耗模型...")
    cpa_trans = CPA(
        power_file='../data/ntt_pipeline_traces_x10k-3rd.txt',
        power_model=power_model_transition,
        key_number=8,
        plaintext_number=10000,
        sample_number=5000,
        thread_number=10
    )
    best_key_trans, best_corr_trans = cpa_trans.multi_thread_start()
    
    print("\n" + "="*50)
    print("=== 最终CPA攻击结果对比 ===")
    print(f"汉明重量模型: Key {best_key_hw} (0x{best_key_hw:X}), 相关系数 {best_corr_hw:.6f}")
    print(f"状态转换模型: Key {best_key_trans} (0x{best_key_trans:X}), 相关系数 {best_corr_trans:.6f}")
    
    if best_corr_hw > best_corr_trans:
        print(f"\n推荐结果: 密钥低3位 = {best_key_hw} (0x{best_key_hw:X}) [汉明重量模型]")
    else:
        print(f"\n推荐结果: 密钥低3位 = {best_key_trans} (0x{best_key_trans:X}) [状态转换模型]")