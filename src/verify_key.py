#!/usr/bin/env python3
"""
验证密钥分析脚本
用于验证真实密钥1000与CPA攻击结果的关系
"""

import numpy as np

WN = 1729
MODULUS = 3329
NUM_UNKNOWN_BITS = 9
UNKNOWN_POSSIBILITIES = 1 << NUM_UNKNOWN_BITS

def analyze_key(true_key):
    """
    分析真实密钥的特性
    """
    print(f"=== 真实密钥分析 ===")
    print(f"真实密钥 b = {true_key}")
    print(f"二进制表示: {bin(true_key)}")
    print(f"低3位: {true_key & 0x7} (二进制: {bin(true_key & 0x7)})")
    print(f"高9位: {true_key >> 3} (二进制: {bin(true_key >> 3)})")
    print()
    
    return true_key & 0x7, true_key >> 3

def test_power_models(plaintext_samples, true_key_low, true_key_high):
    """
    测试不同密钥猜测的功耗模型输出
    """
    print(f"=== 功耗模型测试 ===")
    
    # 重构真实的完整密钥
    true_key_full = (true_key_high << 3) | true_key_low
    print(f"重构的完整密钥: {true_key_full}")
    
    # 测试几个明文样本
    for plaintext in plaintext_samples[:5]:
        print(f"\n明文 a = {plaintext}")
        
        # 计算真实的NTT运算结果
        true_product = (true_key_full * WN) % MODULUS
        true_result = (plaintext + true_product) % MODULUS
        print(f"真实NTT结果: ({plaintext} + {true_product}) % {MODULUS} = {true_result}")
        print(f"真实结果汉明重量: {bin(true_result).count('1')}")
        
        # 测试所有密钥猜测的平均功耗
        print("\n各密钥猜测的平均汉明重量:")
        for key_guess in range(8):
            total_hw = 0
            for b_high in range(UNKNOWN_POSSIBILITIES):
                b_full_guess = (b_high << 3) | key_guess
                product_guess = (b_full_guess * WN) % MODULUS
                result = (plaintext + product_guess) % MODULUS
                total_hw += bin(result).count('1')
            avg_hw = total_hw / UNKNOWN_POSSIBILITIES
            marker = " ← 真实密钥" if key_guess == true_key_low else ""
            print(f"  Key {key_guess}: {avg_hw:.4f}{marker}")

def analyze_cpa_results():
    """
    分析CPA攻击结果
    """
    print(f"\n=== CPA攻击结果分析 ===")
    
    # 从最新的攻击结果
    correlations = {
        0: 0.205421,
        1: 0.026234, 
        2: 0.060738,
        3: 0.089745,
        4: 0.236289,
        5: 0.077332,
        6: 0.058623,
        7: 0.075585
    }
    
    print("相关系数排序:")
    sorted_keys = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    for i, (key, corr) in enumerate(sorted_keys):
        rank = i + 1
        marker = " ← 真实密钥" if key == 0 else ""
        print(f"  {rank}. Key {key}: {corr:.6f}{marker}")
    
    print(f"\n真实密钥 Key 0 排名: {[i+1 for i, (k, _) in enumerate(sorted_keys) if k == 0][0]}")
    print(f"Key 0 与最高相关系数的差距: {max(correlations.values()) - correlations[0]:.6f}")

def read_sample_data(filename, num_samples=10):
    """
    读取样本数据进行分析
    """
    plaintexts = []
    try:
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                if line.strip():
                    plaintext_str = line.split(':', 1)[0]
                    plaintext = int(plaintext_str, 10)
                    plaintexts.append(plaintext)
        print(f"成功读取 {len(plaintexts)} 个明文样本")
    except FileNotFoundError:
        print(f"文件 {filename} 未找到，使用模拟数据")
        plaintexts = [100, 200, 300, 400, 500]  # 模拟数据
    
    return plaintexts

def main():
    print("密钥验证分析工具")
    print("=" * 50)
    
    # 分析真实密钥
    true_key = 1000
    true_key_low, true_key_high = analyze_key(true_key)
    
    # 读取样本数据
    plaintexts = read_sample_data('../data/ntt_pipeline_traces_x10k-3rd.txt')
    
    # 测试功耗模型
    test_power_models(plaintexts, true_key_low, true_key_high)
    
    # 分析CPA结果
    analyze_cpa_results()
    
    print("\n=== 结论 ===")
    print(f"1. 真实密钥的低3位是 {true_key_low}")
    print(f"2. CPA攻击识别出的最佳候选是 Key 4")
    print(f"3. 真实密钥 Key 0 的相关系数为 0.205421，排名第2")
    print(f"4. 可能的原因:")
    print(f"   - 功耗模型不够精确")
    print(f"   - 噪声影响")
    print(f"   - 数据预处理问题")
    print(f"   - NTT运算的功耗特性复杂")
    
if __name__ == "__main__":
    main()