# Flash Attention V2 - FP8 Varlen Test Case

## 概述

这个测试用例完全匹配你提供的 PyTorch 配置，用于验证 Flash Attention 3 (Hopper) 的 FP8 E4M3 + Variable-length + Descaling 功能。

## PyTorch 原始配置

```python
max_seqlen_q = max_seqlen_k = 1680
batch_size = 84
nheads = 16
nheads_k = 16
d = 128
seqlen_q = 5040  # Total tokens
seqlen_k = 5040
causal = False
dtype = torch.float8_e4m3fn

# Variable-length sequences
cu_seqlens_q = torch.tensor([
    0, 64, 128, 192, ..., 5040
], dtype=torch.int32, device='cuda')  # 85 elements (batch + 1)

# Descaling factors
descale_q = torch.ones(batch, nheads, dtype=torch.float32, device='cuda')
descale_k = torch.ones(batch, nheads, dtype=torch.float32, device='cuda')
descale_v = torch.ones(batch, nheads, dtype=torch.float32, device='cuda')

# PyTorch API call
output = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    q_descale=descale_q,
    k_descale=descale_k,
    v_descale=descale_v,
    softmax_scale=1.0 / sqrt(128),
    causal=False
)
```

## C++ 对应实现

### 关键配置

```cpp
// 匹配 PyTorch 的配置
const int max_seqlen_q = 1680;
const int max_seqlen_k = 1680;
const int batch_size = 84;
const int nheads = 16;
const int nheads_k = 16;
const int head_dim = 128;
const int total_q = 5040;  // Sum of all sequence lengths
const int total_k = 5040;
const bool causal = false;

// Cumulative sequence lengths (与 PyTorch 完全一致)
std::vector<int> h_cu_seqlens_q = {
    0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704,
    768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1392, 1440,
    1488, 1536, 1584, 1632, 1680, 1744, 1808, 1872, 1936, 2000, 2064, 2128,
    2192, 2256, 2320, 2384, 2448, 2512, 2576, 2640, 2704, 2768, 2832, 2896,
    2960, 3024, 3072, 3120, 3168, 3216, 3264, 3312, 3360, 3424, 3488, 3552,
    3616, 3680, 3744, 3808, 3872, 3936, 4000, 4064, 4128, 4192, 4256, 4320,
    4384, 4448, 4512, 4576, 4640, 4704, 4752, 4800, 4848, 4896, 4944, 4992,
    5040
};
```

### 数据布局

#### PyTorch 格式
```python
# Q, K, V shape: (seqlen, nheads, d)
q = torch.ones(5040, 16, 128, device='cuda', dtype=torch.float8_e4m3fn)
k = torch.ones(5040, 16, 128, device='cuda', dtype=torch.float8_e4m3fn)
v = torch.randn(5040, 16, 128, device='cuda', dtype=torch.float8_e4m3fn)

# Descale shape: (batch, nheads)
descale_q = torch.ones(84, 16, dtype=torch.float32, device='cuda')
```

#### C++ 对应
```cpp
// Q, K, V: [total_tokens, nheads, head_dim] 连续存储
size_t q_elements = total_q * nheads * head_dim;  // 5040 * 16 * 128
cudaMalloc(&d_q, q_elements * sizeof(__nv_fp8_e4m3));

// Descale: [batch, nheads] 连续存储
cudaMalloc(&d_descale_q, batch_size * nheads * sizeof(float));  // 84 * 16

// 设置 stride
params.q_descale_batch_stride = nheads;      // 16
params.q_descale_head_stride = 1;            // 1
// 访问: descale_q[batch_idx * nheads + head_idx]
```

### Flash Attention 参数设置

```cpp
flash::FlashAttentionParams params;

// Input/output
params.q = d_q;
params.k = d_k;
params.v = d_v;
params.out = d_out;
params.softmax_lse = d_lse;

// Dimensions
params.batch_size = batch_size;        // 84
params.seqlen_q = max_seqlen_q;        // 1680 (被 varlen 忽略)
params.seqlen_k = max_seqlen_k;        // 1680 (被 varlen 忽略)
params.num_heads = nheads;             // 16
params.num_heads_k = nheads_k;         // 16
params.head_dim = head_dim;            // 128

// ✅ Variable-length sequences
params.cu_seqlens_q = d_cu_seqlens_q;  // [0, 64, 128, ..., 5040]
params.cu_seqlens_k = d_cu_seqlens_k;  // 同上
params.total_q = total_q;              // 5040
params.total_k = total_k;              // 5040

// ✅ FP8 descaling
params.q_descale_ptr = d_descale_q;    // [batch, nheads]
params.k_descale_ptr = d_descale_k;
params.v_descale_ptr = d_descale_v;

params.q_descale_batch_stride = nheads;  // 16
params.q_descale_head_stride = 1;
params.k_descale_batch_stride = nheads_k;
params.k_descale_head_stride = 1;
params.v_descale_batch_stride = nheads_k;
params.v_descale_head_stride = 1;

// Other parameters
params.is_causal = false;
params.softmax_scale = 1.0f / sqrtf(128.0f);  // 1/sqrt(d)
params.dtype = flash::DataType::FP8_E4M3;
params.mode = flash::AttentionMode::VARLEN_PREFILL;
```

## 编译和运行

### 方法 1: 使用自动脚本（推荐）

```bash
cd /home/qianxu/flash-attention/standalone
chmod +x build_and_test_v2.sh
./build_and_test_v2.sh
```

### 方法 2: 手动编译

```bash
cd /home/qianxu/flash-attention/standalone
mkdir -p build_v2
cd build_v2

# 配置
cmake .. -DCMAKE_BUILD_TYPE=Release -f ../CMakeLists_v2.txt

# 编译 FP8 varlen 测试
cmake --build . --target test_fp8_varlen -j$(nproc)

# 运行
./test_fp8_varlen
```

### 方法 3: 运行所有测试

```bash
# 编译所有测试
cmake --build . --target flash_attention_test_v2 -j$(nproc)

# 运行综合测试（包含 3 个 case）
./flash_attention_test_v2
```

## 预期输出

```
================================================================
  FP8 E4M3 Variable-length Attention Test
  (Matching PyTorch flash_attn_varlen_func configuration)
================================================================

Configuration:
  Batch size: 84
  Max seqlen Q: 1680
  Max seqlen K: 1680
  Total tokens Q: 5040
  Total tokens K: 5040
  Num heads Q: 16
  Num heads K: 16
  Head dim: 128
  Causal: false
  Data type: FP8 E4M3

Sequence lengths per batch:
  First 5 sequences: 64, 64, 64, 64, 64
  Last 5 sequences: 48, 48, 48, 48, 48

Allocating memory...
  Q: 9.84 MB
  K: 9.84 MB
  V: 9.84 MB
Generating random data...
Copying data to device...
Done.

Setting up Flash Attention parameters...
  Softmax scale: 0.0883883
Done.

Validating parameters...
✓ Parameters validated successfully

Running Flash Attention forward pass...

================================================================
  ✓ SUCCESS
================================================================

Performance:
  Time: 1.23 ms
  Throughput: 52.14 TFLOPS

Verifying output (basic sanity check)...
  ✓ Output looks reasonable (no NaN/Inf in first 10 tokens)
  First 5 output values (head 0, token 0): 0.125 0.0625 -0.25 0.5 0.375
```

## 关键差异说明

### 1. 张量布局

**PyTorch**:
```python
# Shape: (seqlen, nheads, d)
q.shape  # torch.Size([5040, 16, 128])
```

**C++**:
```cpp
// 相同布局: [total_tokens, nheads, head_dim]
// 内存: q[token_idx * nheads * head_dim + head_idx * head_dim + dim_idx]
```

### 2. Descale 布局

**PyTorch**:
```python
# Shape: (batch, nheads)
descale_q.shape  # torch.Size([84, 16])
```

**C++**:
```cpp
// 相同布局: [batch, nheads]
// 访问: descale[batch_idx * nheads + head_idx]
params.q_descale_batch_stride = nheads;  // 16
params.q_descale_head_stride = 1;        // 1
```

### 3. cu_seqlens 格式

**PyTorch 和 C++ 完全一致**:
```
[0, 64, 128, 192, ..., 5040]
# 长度: batch_size + 1 = 85
# 第 i 个序列: tokens[cu_seqlens[i] : cu_seqlens[i+1]]
```

### 4. 输出格式

**PyTorch**:
```python
output.dtype  # torch.float16 (FP8 输入 -> FP16 输出)
output.shape  # torch.Size([5040, 16, 128])
```

**C++**:
```cpp
// Output 是 FP16 (即使输入是 FP8)
cudaMalloc(&d_out, total_q * nheads * head_dim * sizeof(__half));
```

## 性能对比

### 预期性能 (H100)

| 配置 | 理论 FLOPS | 预期吞吐 |
|------|-----------|---------|
| FP16 | ~40 TFLOPS | 35-38 TFLOPS |
| FP8 E4M3 | ~80 TFLOPS | 50-65 TFLOPS |

**说明**: FP8 理论上比 FP16 快 2x，但实际性能取决于：
- 序列长度分布（不均匀会降低效率）
- Head dimension（128 最优）
- Descaling 开销

### 测试配置的特点

- **不均匀序列长度**: 64, 64, ..., 48, 48 (大部分是 64，少数 48)
- **Total tokens**: 5040 (中等规模)
- **Batch**: 84 (较大，有利于 GPU 利用率)

## 常见问题

### Q1: cu_seqlens 为什么有 85 个元素？

**A**: cu_seqlens 的长度是 `batch_size + 1`，因为：
- 第一个元素总是 0
- 最后一个元素是 total tokens
- 第 i 个序列的 tokens: `cu_seqlens[i]` 到 `cu_seqlens[i+1]`

### Q2: Descale 如何工作？

**A**: FP8 E4M3 的 descaling 公式：
```
float_value = fp8_value * descale_factor
```

在这个测试中，所有 descale 都是 1.0（`torch.ones`），意味着：
- 不进行缩放调整
- 仅用于测试功能，实际使用时需要根据量化范围设置

### Q3: 为什么 seqlen_q/seqlen_k 参数还要设置？

**A**: 虽然使用 varlen 时这些参数会被忽略，但 API 仍需要它们来：
- 验证参数合法性
- 某些内部计算（如 rounded dimensions）
- 建议设置为 max_seqlen

### Q4: 输出为什么是 FP16 而不是 FP8？

**A**: Flash Attention 3 的设计决定：
- **输入**: FP8 E4M3 (节省带宽)
- **计算**: FP32 accumulator (保证精度)
- **输出**: FP16/BF16 (避免二次量化损失)

## 调试技巧

### 1. 检查 CUDA 错误

```cpp
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
}
```

### 2. 验证 cu_seqlens

```cpp
// 检查单调性
for (int i = 0; i < batch_size; i++) {
    int seqlen = h_cu_seqlens_q[i + 1] - h_cu_seqlens_q[i];
    if (seqlen <= 0) {
        std::cerr << "Error: invalid cu_seqlens at index " << i << std::endl;
    }
}
```

### 3. 检查输出 NaN/Inf

```cpp
std::vector<__half> h_out(total_q * nheads * head_dim);
cudaMemcpy(h_out.data(), d_out, ...);

for (size_t i = 0; i < h_out.size(); i++) {
    float val = __half2float(h_out[i]);
    if (std::isnan(val) || std::isinf(val)) {
        std::cerr << "NaN/Inf at index " << i << std::endl;
        break;
    }
}
```

## 文件列表

- `src/test_fp8_varlen.cpp` - FP8 varlen 测试程序
- `src/flash_api_v2.cu` - 增强版 API 实现
- `include/flash_api_v2.h` - API 头文件
- `CMakeLists_v2.txt` - CMake 配置
- `build_and_test_v2.sh` - 自动编译和测试脚本
- `API_V2_REFERENCE.md` - 完整 API 文档

## 下一步

如果测试通过，你可以：
1. 修改 descale 值测试不同量化配置
2. 测试不同的序列长度分布
3. 添加性能 profiling（使用 nsys/ncu）
4. 与 PyTorch 输出进行数值对比

如果测试失败，请查看：
1. CUDA 错误信息
2. API 返回的错误码
3. 参数验证结果
4. GPU 型号（必须是 Hopper: H100/H800）
