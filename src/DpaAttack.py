import os
import queue
import threading as td
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.artist import kwdoc

import numpy as np

WN = 1729
MODULUS = 3329
# Unknown bit number
NUM_UNKNOWN_BITS = 9  # 12-bit key - 3-bit guess = 9 unknown bits
UNKNOWN_POSSIBILITIES = 1 << NUM_UNKNOWN_BITS # 2^9 = 512

def V_internal(plaintext, key_low_guess):
    """
    一个适用于部分密钥猜测的DPA T-Test分类函数.
    它通过平均掉未知比特的影响来决定分类.

    Args:
        plaintext (int): 输入的明文 a.
        key_low_guess (int): 对密钥低3位的猜测 (0-7).

    Returns:
        int: 0 或 1，用于DPA分类.
    """
    
    predictions_true_count = 0

    # 遍历未知高位的所有可能性 (0 to 511)
    for b_high in range(UNKNOWN_POSSIBILITIES):
        # 1. 重构一个完整的12位b的猜测值
        #    将高9位和我们假设的低3位拼接起来
        b_full_guess = (b_high << 3) | key_low_guess
        # 2. 根据完整的b_guess计算product
        product_guess = (b_full_guess * WN) % MODULUS
        
        # 3. 判断条件是否为真
        if (plaintext + product_guess) >= MODULUS:
            predictions_true_count += 1
            
    # 4. 计算条件为真的概率
    probability_true = predictions_true_count / UNKNOWN_POSSIBILITIES
    
    # 5. 根据概率进行二元分类
    #    如果概率大于50%，我们认为它更倾向于“发生减法”的高功耗事件，归为1类
    if probability_true > 0.5:
        return 1
    else:
        return 0

# def V_internal(plaintext, key_guess):
#     # V = 1 if  plaintext+(key_guess*1729)%3329 > 3329 else 0
#     # return V
#     V = plaintext+((key_guess<<3)*1729)%3329
#     return V  & 0x01


class DPA:
    def __init__(self,
                 power_file,
                 V_interval,
                 key_number,
                 plaintext_number=4096,
                 sample_number=5000,
                 thread_number=10):
        # class
        self.power_file = power_file
        self.V_interval = V_interval
        self.key_number = key_number
        self.plaintext_number = plaintext_number
        self.sample_number = sample_number
        self.thread_number = thread_number
        self.threads = {}
        self.q = queue.Queue()

        #dpa power set
        self.set_0 = [np.zeros(sample_number, dtype=np.int64) for _ in range(key_number)]
        self.set_1 = [np.zeros(sample_number, dtype=np.int64) for _ in range(key_number)]
        self.set_0_num = [0] * key_number
        self.set_1_num = [0] * key_number
        #result
        self.power_list = [np.zeros(sample_number) for _ in range(key_number)]
        self.time = list(range(sample_number))

        self.counter_lock = td.Lock()
        self.key_locks = [td.Lock() for _ in range(key_number)]

    def thread_read_signals(self):
        print(f'thread power trace read start')
        number = 0
        with open(self.power_file, 'r') as pf:
            for line in pf:
                if ( number >= self.plaintext_number ) or ( not line.strip() ):
                    break
                try:
                    plaintext_str, power_trace_str = line.split(':', 1)
                    plaintext = int(plaintext_str)
                    power_trace = np.array(power_trace_str.strip().split()).astype(np.int64)

                    if len(power_trace) < self.sample_number:
                        power_trace = np.pad(power_trace, (0, self.sample_number - len(power_trace)))
                    elif len(power_trace) > self.sample_number:
                        power_trace = power_trace[:self.sample_number]
                    #message
                    signal_data = {'plaintext': plaintext,'power_trace': power_trace}
                    self.q.put(signal_data)

                    print(f'read add {plaintext}')
                    number += 1
                except Exception as e:
                    print(f"Error parsing line: {line.strip()} - {str(e)}")
        #threads exit signal
        for _ in range(self.thread_number):
            self.q.put({'plaintext': 'EXIT_SIGNAL', 'power_trace': None})
        print(f'thread read end, added {number} traces')

    def thread_signals_analyze(self, thread_id):
        print(f'thread analyze {thread_id} start')
        processed_count = 0
        while True:
            try:
                signal = self.q.get(timeout=10)  # 添加超时避免无限等待
                if signal['plaintext'] == 'EXIT_SIGNAL':
                    print(f'thread {thread_id} received exit signal')
                    self.q.task_done()
                    break
                plaintext = signal['plaintext']
                power_trace = signal['power_trace']
                print(f'thread {thread_id} get data {plaintext}')
                for key in range(self.key_number):
                    try:
                        V = self.V_interval(plaintext, key)
                        with self.key_locks[key]:
                            if V == 0:
                                self.set_0[key] += power_trace
                                with self.counter_lock:
                                    self.set_0_num[key] += 1
                                print(f'thread {thread_id} add set0 data for key {key}')
                            else:
                                self.set_1[key] += power_trace
                                with self.counter_lock:
                                    self.set_1_num[key] += 1
                                print(f'thread {thread_id} add set1 data for key {key}')
                    except Exception as e:
                        print(f"Thread {thread_id} key {key} error: {str(e)}")
                processed_count += 1
                self.q.task_done()
            except queue.Empty:
                print(f"Thread {thread_id} queue empty, exiting")
                break
            except Exception as e:
                print(f"Thread {thread_id} general error: {str(e)}")
        print(f'thread analyze {thread_id} end, processed {processed_count} traces')

    def get_delta_power(self):
        for key in range(self.key_number):
            try:
                set0_count = self.set_0_num[key]
                set1_count = self.set_1_num[key]
                # 计算平均值
                if set0_count > 0:
                    set_0_mean = self.set_0[key] / set0_count
                else:
                    set_0_mean = np.zeros(self.sample_number)

                if set1_count > 0:
                    set_1_mean = self.set_1[key] / set1_count
                else:
                    set_1_mean = np.zeros(self.sample_number)

                self.power_list[key] = set_0_mean - set_1_mean
                # print(f"\nKey {key} stats:")
                # print(f"  Set 0: count={set0_count}, sum={self.set_0[key]}")
                # print(f"  Set 0: average={set_0_mean}")
                # print(f"  Set 1: count={set1_count}, sum={self.set_1[key]}")
                # print(f"  Set 1: average={set_1_mean}")
                # print(f"  Delta power: {self.power_list[key][:5]}")

            except Exception as e:
                print(f"Error calculating delta for key {key}: {str(e)}")
                self.power_list[key] = np.zeros(self.sample_number)

        return self.power_list

    def multi_thread_start(self):
        print(f'start main process: {os.getpid()}')

        reader = td.Thread(target=self.thread_read_signals)
        reader.start()
        self.threads['reader'] = reader

        analyzers = []
        for thread_id in range(self.thread_number):
            analyzer = td.Thread(
                target=self.thread_signals_analyze,
                args=(thread_id,)
            )
            analyzer.daemon = True  # 设置为守护线程
            analyzer.start()
            analyzers.append(analyzer)
            self.threads[thread_id] = analyzer

        reader.join()
        print("Reader thread completed")
        self.q.join()
        print("All queue tasks completed")

        for analyzer in analyzers:
            analyzer.join(timeout=5)



        print("\n所有线程已完成")

        self.result_analyze()
        print(f'end main process: {os.getpid()}')

    def result_analyze(self):
        result = self.get_delta_power()
        fig1, axs1 = plt.subplots(4, 1, figsize=(10, 8))  # 2行2列
        fig2, axs2 = plt.subplots(4, 1, figsize=(10, 8))  # 2行2列

        print("差分功耗分析结果:")

        # for key, delta in enumerate(result):
        #     print(f"Key {key}: {delta[:5]}...")
        #     if key <=3 :
        #         axs1[key].plot(self.time,delta)
        #         axs1[key].set_title(f"key(hex): {hex(key)}")
        #     else :
        #         axs2[key-4].plot(self.time, delta)
        #         axs2[key-4].set_title(f"key(hex): {hex(key)}")
        # plt.tight_layout()  # 自动调整间距

        max_power =0
        max_key = 0
        for key, delta in enumerate(result):
            d_max = np.max(np.abs(result[key]))
            if d_max > max_power :
                print(f'add key : {key}, power {d_max}')
                max_key = key
                max_power = d_max
        print(f'max key : {max_key}, max power : {max_power}')
        # plt.show()



if __name__ == "__main__":

    dpa = DPA(
        power_file='../data/ntt_pipeline_traces_x10k-3rd.txt',
        V_interval=V_internal,
        key_number=8,
        plaintext_number=10000,
        sample_number=5000,
        thread_number=30
    )
    dpa.multi_thread_start()