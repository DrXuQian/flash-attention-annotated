# 检查输入数据一致性

如果kernel参数完全一致，但性能不同，很可能是**输入数据不同**。

## 为什么数据会影响性能？

### 1. **缓存命中率**
- 全零数据：高缓存压缩率，更快
- 随机数据：低缓存压缩率，更慢
- 重复pattern：可能被prefetcher预测，更快

### 2. **分支预测**
Flash Attention内部有很多条件判断：
```cpp
// 例子1: Causal mask
if (m < n) skip;  // 上三角跳过

// 例子2: Softmax优化
if (max_score < threshold) skip_tile;

// 例子3: Early exit
if (attention_weight < epsilon) continue;
```

**不同的Q/K/V值会导致不同的分支路径！**

### 3. **数值精度影响**
```cpp
// FP16有特殊值：
// - 0.0: 特殊优化路径
// - INF/-INF: 触发特殊处理
// - NaN: 检查和处理
// - Subnormal: 可能更慢
```

### 4. **Tile计算跳过**
```cpp
// Flash Attention可能跳过某些tiles
if (this_tile_all_masked) {
    // 直接跳过，不计算
    continue;
}
```

---

## 如何检查数据一致性

### 方法1：在kernel launch前打印统计信息

在 `hopper/flash_fwd_launch_template.h` 中添加：

```cpp
#include "../standalone/include/debug_input_data.h"

void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    #ifdef FLASH_DEBUG_INPUT_DATA
    PRINT_INPUT_TENSOR_STATS(params);
    #endif

    // ... rest of function
}
```

编译：
```bash
nvcc -DFLASH_DEBUG_INPUT_DATA ...
```

**输出示例：**
```
========== INPUT TENSOR STATISTICS ==========
[TENSOR STATS] Q (size=2099200):
  min=-3.142578, max=3.140625, mean=0.000123, std=1.001234
  sum=258.456, checksum_abs_sum=258.456
[TENSOR STATS] K (size=2099200):
  min=-2.998047, max=2.996094, mean=-0.000456, std=0.998765
  sum=-957.123, checksum_abs_sum=957.123
[TENSOR STATS] V (size=2099200):
  min=-3.001953, max=3.003906, mean=0.000789, std=1.002345
  sum=1656.789, checksum_abs_sum=1656.789
=============================================
```

**对比PyTorch和Standalone的输出：**
- `min/max` 应该完全一致
- `mean/std` 应该非常接近（可能有小的浮点误差）
- `sum` 应该非常接近

如果不一致 → **数据不同！**

---

### 方法2：保存输入数据到文件对比

**在PyTorch中：**
```python
import torch
import numpy as np

# 保存Q/K/V到文件
q.cpu().numpy().tofile('q_pytorch.bin')
k.cpu().numpy().tofile('k_pytorch.bin')
v.cpu().numpy().tofile('v_pytorch.bin')
```

**在Standalone中：**
```cpp
// 在test文件中，初始化Q/K/V后
FILE* f = fopen("q_standalone.bin", "wb");
cudaMemcpy(h_q.data(), d_q, q_size, cudaMemcpyDeviceToHost);
fwrite(h_q.data(), sizeof(__half), q_elements, f);
fclose(f);
```

**对比：**
```bash
cd standalone/scripts
nvcc -o check_input_data check_input_data.cu
./check_input_data q_pytorch.bin q_standalone.bin 2099200
```

---

### 方法3：使用相同的seed初始化

确保PyTorch和Standalone使用**完全相同的随机数种子**：

**PyTorch:**
```python
torch.manual_seed(42)
torch.cuda.manual_seed(42)
q = torch.randn(..., generator=torch.Generator().manual_seed(42))
```

**Standalone:**
```cpp
#include <curand_kernel.h>

// 使用相同的seed
srand(42);
// 或者使用curand with same seed
curandGenerator_t gen;
curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
curandSetPseudoRandomGeneratorSeed(gen, 42);
```

---

## 实际case：数据分布差异

### Case 1: 初始化方法不同

**PyTorch:**
```python
q = torch.randn(...)  # 标准正态分布 N(0,1)
```

**Standalone (错误):**
```cpp
// 错误：均匀分布！
for (int i = 0; i < size; i++) {
    h_q[i] = __float2half((rand() / (float)RAND_MAX) * 2 - 1);  // U(-1,1)
}
```

这会导致：
- 不同的mean/std
- 不同的min/max
- 不同的性能特征！

### Case 2: 缩放因子不同

**PyTorch:**
```python
q = torch.randn(...) * 0.1  # 缩放到更小范围
```

**Standalone:**
```cpp
// 忘记缩放
q = randn();  // 没有 * 0.1
```

虽然分布相同，但数值范围不同，影响：
- Softmax的exp计算（大数值可能overflow）
- 数值精度（FP16范围有限）

---

## 快速checklist

对比以下内容：

### 数据统计：
- [ ] Q的min/max/mean/std
- [ ] K的min/max/mean/std
- [ ] V的min/max/mean/std
- [ ] Descale factors (if FP8)

### 初始化方法：
- [ ] Random seed相同
- [ ] 分布类型相同（正态 vs 均匀）
- [ ] 缩放因子相同
- [ ] 数据类型相同（FP16 vs BF16 vs FP32）

### 特殊值：
- [ ] 是否有NaN？
- [ ] 是否有INF？
- [ ] 是否有Subnormal数？
- [ ] 零值比例

---

## 如果数据完全一致但性能仍不同

可能的原因：

### 1. **GPU状态不同**
```bash
# 检查GPU频率
nvidia-smi -q -d CLOCK

# 检查温度
nvidia-smi -q -d TEMPERATURE

# 检查功耗模式
nvidia-smi -q -d POWER
```

### 2. **CUDA context初始化**
第一次调用vs后续调用性能不同：
```cpp
// Warm up
for (int i = 0; i < 3; i++) {
    flash_attention_forward(params, stream);
}
cudaDeviceSynchronize();

// 然后再测量
auto start = std::chrono::high_resolution_clock::now();
flash_attention_forward(params, stream);
cudaDeviceSynchronize();
auto end = std::chrono::high_resolution_clock::now();
```

### 3. **测量方法不同**
```cpp
// 方法1: CUDA event (推荐)
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start);
kernel<<<...>>>();
cudaEventRecord(stop);
cudaEventSynchronize(stop);
float ms;
cudaEventElapsedTime(&ms, start, stop);

// 方法2: CPU时间 (不准确！包含CPU overhead)
auto t1 = std::chrono::high_resolution_clock::now();
kernel<<<...>>>();
cudaDeviceSynchronize();
auto t2 = std::chrono::high_resolution_clock::now();
```

### 4. **编译优化差异**

检查是否都使用了：
```cmake
-O3
--use_fast_math
-DNDEBUG
-gencode arch=compute_90a,code=sm_90a
```

---

## Summary

**调试顺序：**
1. ✅ Kernel参数一致
2. ✅ Template参数一致
3. ⬅️ **你在这里：检查输入数据**
4. ❓ GPU状态
5. ❓ 测量方法
6. ❓ 编译选项

使用 `PRINT_INPUT_TENSOR_STATS()` 快速验证数据是否一致！
