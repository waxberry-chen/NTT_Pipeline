# 侧信道攻击

## 〇. 现有的攻击方式

### 0.1 测试, 计算过程

首先第一次测试视为"构造模板" (我感觉这个不能算作模板攻击), 经过降维得到一个1500*3329的矩阵. 
$$
Template =
\begin{pmatrix}
O_{0,0} & O_{0,1} & \dots & O_{0,j} & \dots & O_{0,3328} \\
O_{1,0} & O_{1,1} & \dots & O_{1,j} & \dots & O_{1,3328} \\
\vdots & \vdots & \ddots & \vdots & \dots & \vdots \\ 
O_{i,0} & O_{i,1} & \dots & O_{i,j} & \dots & O_{i,3328} \\
\vdots & \vdots & \dots & \vdots & \ddots & \vdots \\ 
O_{1499,0} & O_{1499,1} & \dots & O_{1499,j} & \dots & O_{1499,3328} \\
\end{pmatrix}
$$
