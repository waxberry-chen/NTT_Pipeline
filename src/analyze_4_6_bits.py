#!/usr/bin/env python3
"""
分析4-6位攻击结果，验证真实密钥计算和攻击失败原因
"""

# 真实密钥分析
true_key = 1000
print(f"真实密钥: {true_key}")
print(f"二进制表示: {true_key:012b}")
print()

# 位分解
low_3_bits = true_key & 0x7  # 低3位 (0-2)
bits_4_6 = (true_key >> 3) & 0x7  # 4-6位 (3-5)
high_bits = true_key >> 6  # 高位 (6-11)

print("位分解:")
print(f"低3位 (0-2): {low_3_bits} (二进制: {low_3_bits:03b})")
print(f"4-6位 (3-5): {bits_4_6} (二进制: {bits_4_6:03b})")
print(f"高位 (6-11): {high_bits} (二进制: {high_bits:06b})")
print()

# 验证重构
reconstructed = (high_bits << 6) | (bits_4_6 << 3) | low_3_bits
print(f"重构验证: {reconstructed} (应该等于 {true_key})")
print(f"重构正确: {'✓' if reconstructed == true_key else '✗'}")
print()

# 分析攻击结果
print("=== 攻击结果分析 ===")
print("从攻击输出可以看到:")
print("- 位转换模型推荐: 4 (100), 相关系数: 0.532184")
print("- 真实4-6位: 5 (101), 相关系数: 0.527169")
print("- 差距很小: 0.532184 - 0.527169 = 0.005015")
print()

print("可能的原因:")
print("1. 噪声影响: 相关系数差距很小，可能是噪声导致的")
print("2. 功耗模型不够精确: 需要更准确地建模4-6位的功耗特性")
print("3. 样本数量: 可能需要更多样本来提高统计显著性")
print("4. 位间相关性: 4-6位可能与其他位有强相关性")
print()

# 分析4和5的二进制差异
print("=== 候选4和真实5的差异分析 ===")
candidate_4 = 4  # 100
true_5 = 5       # 101
print(f"候选4: {candidate_4:03b}")
print(f"真实5: {true_5:03b}")
print(f"汉明距离: {bin(candidate_4 ^ true_5).count('1')}")
print("差异: 只有最低位不同 (第3位)")
print()

# 建议改进方案
print("=== 改进建议 ===")
print("1. 增加样本数量到20000+")
print("2. 使用更精确的功耗模型")
print("3. 考虑多位联合攻击")
print("4. 分析功耗迹线的质量")
print("5. 尝试不同的统计方法 (如互信息)")

# 检查进位传播模型的表现
print("\n=== 进位传播模型分析 ===")
print("进位传播模型表现最好:")
print("- 推荐: 2 (010), 相关系数: 0.819843")
print("- 真实5: 相关系数: 0.808179")
print("- 差距: 0.011664")
print("这表明进位传播在4-6位攻击中很重要，但模型仍需优化")