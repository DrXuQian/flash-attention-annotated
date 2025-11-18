# Flash Attention V2 API Reference

## Overview

这是一个增强版的 Flash Attention 3 (Hopper) standalone API，明确标注了所有支持和不支持的特性。

## 支持的配置 (Supported Configurations)

### ✅ 1. FP16 Prefill with Causal Mask

**用途**: 自回归模型的训练/预填充（如 GPT）

```cpp
flash::FlashAttentionParams params;
params.q = d_q;  // [batch, seqlen, num_heads, head_dim]
params.k = d_k;
params.v = d_v;
params.out = d_out;

params.batch_size = 2;
params.seqlen_q = 512;
params.seqlen_k = 512;
params.num_heads = 16;
params.num_heads_k = 16;  // MHA
params.head_dim = 128;

params.is_causal = true;  // ✅ Causal masking
params.dtype = flash::DataType::FP16;

flash::flash_attention_forward(params, stream);
```

**特性**:
- ✅ Causal masking (lower triangular mask)
- ✅ 所有 head dimensions: 64, 96, 128, 192, 256
- ✅ MHA (Multi-Head Attention)
- ✅ GQA (Grouped Query Attention)
- ✅ MQA (Multi-Query Attention, num_heads_k=1)

---

### ✅ 2. FP16 Decoding (Inference)

**用途**: 推理时的单 token 生成（seqlen_q = 1）

```cpp
flash::FlashAttentionParams params;
params.q = d_q;  // [batch, 1, num_heads, head_dim] - 单个新 token
params.k = d_k;  // [batch, context_len, num_heads_k, head_dim] - 完整上下文
params.v = d_v;
params.out = d_out;

params.batch_size = 4;
params.seqlen_q = 1;           // ✅ Single token
params.seqlen_k = 2048;        // Full context length
params.num_heads = 16;
params.num_heads_k = 2;        // ✅ GQA (16:2 ratio, like Qwen2.5-VL)
params.head_dim = 128;

params.is_causal = false;      // 不需要 causal（已经解码到当前位置）
params.dtype = flash::DataType::FP16;
params.mode = flash::AttentionMode::DECODE;

flash::flash_attention_forward(params, stream);
```

**特性**:
- ✅ seqlen_q = 1 (single token generation)
- ✅ GQA/MQA support (节省 KV cache 内存)
- ✅ Large context lengths (tested up to 32K+)
- ⚠️ **不支持** Paged KV cache（需要额外的 kernel 实例化）

---

### ✅ 3. FP8 E4M3 with Variable-length Sequences + Descaling

**用途**: 训练时的变长序列批处理 + FP8 量化

```cpp
// 假设 batch 中有 3 个序列，长度分别为 [256, 512, 128]
int batch_size = 3;
std::vector<int> seqlens = {256, 512, 128};

// 计算累积长度
std::vector<int> h_cu_seqlens(batch_size + 1);
h_cu_seqlens[0] = 0;
for (int i = 0; i < batch_size; i++) {
    h_cu_seqlens[i + 1] = h_cu_seqlens[i] + seqlens[i];
}
int total_tokens = h_cu_seqlens[batch_size];  // 896

// Q/K/V 张量是连续的，不按 batch 分隔
// Shape: [total_tokens, num_heads, head_dim]
void* d_q;  // FP8 E4M3 data
void* d_k;
void* d_v;
cudaMalloc(&d_q, total_tokens * num_heads * head_dim * sizeof(__nv_fp8_e4m3));
// ... (同样 K, V)

// 分配 cu_seqlens
int* d_cu_seqlens_q, *d_cu_seqlens_k;
cudaMalloc(&d_cu_seqlens_q, (batch_size + 1) * sizeof(int));
cudaMalloc(&d_cu_seqlens_k, (batch_size + 1) * sizeof(int));
cudaMemcpy(d_cu_seqlens_q, h_cu_seqlens.data(), ...);
cudaMemcpy(d_cu_seqlens_k, h_cu_seqlens.data(), ...);

// FP8 descale factors (全局或 per-head)
float* d_q_descale, *d_k_descale, *d_v_descale;
cudaMalloc(&d_q_descale, sizeof(float));  // 单个全局 scale
cudaMalloc(&d_k_descale, sizeof(float));
cudaMalloc(&d_v_descale, sizeof(float));

float descale_value = 0.1f;  // 示例值
cudaMemcpy(d_q_descale, &descale_value, sizeof(float), cudaMemcpyHostToDevice);
// ... (同样 K, V)

flash::FlashAttentionParams params;
params.q = d_q;
params.k = d_k;
params.v = d_v;
params.out = d_out;  // Output 是 FP16/BF16

params.batch_size = batch_size;
params.seqlen_q = 0;  // ⚠️ 使用 varlen 时被忽略
params.seqlen_k = 0;  // ⚠️ 使用 varlen 时被忽略
params.num_heads = 16;
params.num_heads_k = 16;
params.head_dim = 128;

// ✅ Variable-length sequences
params.cu_seqlens_q = d_cu_seqlens_q;
params.cu_seqlens_k = d_cu_seqlens_k;
params.total_q = total_tokens;
params.total_k = total_tokens;

// ✅ FP8 descaling
params.q_descale_ptr = d_q_descale;
params.k_descale_ptr = d_k_descale;
params.v_descale_ptr = d_v_descale;
params.q_descale_batch_stride = 0;  // 0 表示全局 scaling
params.q_descale_head_stride = 0;
// ... (同样 K, V)

params.dtype = flash::DataType::FP8_E4M3;
params.mode = flash::AttentionMode::VARLEN_PREFILL;

flash::flash_attention_forward(params, stream);
```

**FP8 Descaling 说明**:

FP8 E4M3 格式的动态范围有限（~[-448, 448]），需要 descaling 来恢复原始值：

```
float_value = fp8_value * descale_factor
```

支持两种 descaling 模式：

1. **全局 scaling** (所有 heads 共享一个 scale):
   ```cpp
   params.q_descale_batch_stride = 0;
   params.q_descale_head_stride = 0;
   // d_q_descale 只需要 1 个 float
   ```

2. **Per-head scaling** (每个 head 独立 scale):
   ```cpp
   params.q_descale_batch_stride = num_heads;
   params.q_descale_head_stride = 1;
   // d_q_descale 需要 [batch, num_heads] 个 float
   ```

**特性**:
- ✅ FP8 E4M3 quantization
- ✅ Variable-length sequences (via `cu_seqlens`)
- ✅ Descaling (global or per-head)
- ✅ Head dimensions: 64, 96, 128 (FP8 不支持 192, 256)
- ⚠️ Output 总是 FP16/BF16 (不是 FP8)

---

### ✅ 4. FP16 Variable-length Prefill

**用途**: 训练时的变长序列（不用 padding，节省计算）

```cpp
flash::FlashAttentionParams params;
// ... (设置同 FP8 varlen，但 dtype = FP16)

params.cu_seqlens_q = d_cu_seqlens_q;
params.cu_seqlens_k = d_cu_seqlens_k;
params.total_q = total_tokens;
params.total_k = total_tokens;

params.dtype = flash::DataType::FP16;  // FP16 instead of FP8
// 不需要 descale pointers

flash::flash_attention_forward(params, stream);
```

---

## 不支持的特性 (Unsupported Features)

### ❌ 1. BF16 数据类型
```cpp
params.dtype = flash::DataType::BF16;  // ❌ NOT SUPPORTED
```
**原因**: 需要编译 BF16 kernel 实例化（未包含在 standalone build 中）

---

### ❌ 2. Softcapping (Gemini/Gemma)
```cpp
params.softcap = 30.0f;  // ❌ NOT SUPPORTED
```
**原因**: 需要 `Has_softcap=true` 的 kernel 实例化

**什么是 Softcapping**:
```
score = softcap * tanh(score / softcap)
```
用于限制 attention scores 的范围，防止极端值（Gemini/Gemma 模型使用）

---

### ❌ 3. PackGQA 优化
```cpp
params.pack_gqa = true;  // ❌ NOT SUPPORTED
```
**原因**: 需要 `PackGQA=true` 的 kernel 实例化

**什么是 PackGQA**:
一种内存布局优化，将 Q heads 按 KV head 分组打包，减少内存访问。例如：
- 普通 GQA: `Q[batch, seqlen, 16 heads, headdim]`
- PackGQA: `Q[batch, seqlen, 2 kv_groups, 8 q_per_kv, headdim]`

---

### ❌ 4. Split-KV (长序列分块)
```cpp
params.num_splits = 4;  // ❌ MUST BE 1
```
**原因**: 需要 `Split=true` 的 kernel 实例化 + combine kernel

**什么是 Split-KV**:
将 K/V 沿序列维度分成多块并行计算，最后合并结果。用于超长序列（>64K tokens）。

---

### ❌ 5. Paged KV Cache
```cpp
params.page_table = d_page_table;  // ❌ NOT SUPPORTED
params.page_size = 64;
```
**原因**: 需要 `PagedKVNonTMA=true` 或 `pagedkv_tma=true` 的 kernel 实例化

**什么是 Paged KV Cache**:
像虚拟内存一样，将 KV cache 分成固定大小的 pages，支持动态分配和共享。用于 serving 系统（如 vLLM）。

---

### ❌ 6. Rotary Position Embeddings (RoPE)
```cpp
params.rotary_cos = d_rotary_cos;  // ❌ NOT SUPPORTED
params.rotary_sin = d_rotary_sin;
params.rotary_dim = 64;
```
**原因**: 需要额外的 RoPE kernel 逻辑

**解决方案**: 在调用 Flash Attention **之前**，手动对 Q 和 K 应用 RoPE。

---

### ❌ 7. Attention Dropout
```cpp
params.p_dropout = 0.1f;  // ❌ MUST BE 0.0
params.rng_state = d_rng_state;
```
**原因**: 需要 RNG 状态管理和额外的 kernel 逻辑

**解决方案**: 在 Flash Attention 之后手动应用 dropout。

---

### ❌ 8. Custom Attention Mask
```cpp
params.attn_mask = d_custom_mask;  // ❌ NOT SUPPORTED
```
**支持的 masking**:
- ✅ Causal mask (`is_causal = true`)
- ✅ Local/sliding window (`window_size_left/right`)
- ❌ 任意的自定义 mask

---

### ❌ 9. Left Padding for K/V
```cpp
params.leftpad_k = d_leftpad_indices;  // ❌ NOT SUPPORTED
```
**解决方案**: 使用 `seqused_k` 参数来指定实际序列长度。

---

### ❌ 10. Appending New KV
```cpp
params.k_new = d_k_new;  // ❌ NOT SUPPORTED
params.v_new = d_v_new;
params.seqlen_knew = 128;
```
**用途**: 在现有 KV cache 后追加新的 K/V（用于 prefill + append）

**解决方案**: 手动拼接 K 和 K_new，然后调用 Flash Attention。

---

## Kernel 实例化列表

当前 standalone build 包含以下 kernel 实例化：

```cpp
// FP16 kernels (所有 head dimensions)
run_mha_fwd_<90, cutlass::half_t, 64, 64, false, false, false, false>
run_mha_fwd_<90, cutlass::half_t, 96, 96, false, false, false, false>
run_mha_fwd_<90, cutlass::half_t, 128, 128, false, false, false, false>
run_mha_fwd_<90, cutlass::half_t, 192, 192, false, false, false, false>
run_mha_fwd_<90, cutlass::half_t, 256, 256, false, false, false, false>

// FP8 E4M3 kernels (limited head dimensions)
run_mha_fwd_<90, cutlass::float_e4m3_t, 64, 64, false, false, false, false>
run_mha_fwd_<90, cutlass::float_e4m3_t, 96, 96, false, false, false, false>
run_mha_fwd_<90, cutlass::float_e4m3_t, 128, 128, false, false, false, false>
```

**模板参数说明**:
```cpp
run_mha_fwd_<Arch, DType, HeadDim, HeadDimV, Split, PagedKVNonTMA, Has_softcap, PackGQA>
//           90    half_t  128     128       false  false          false        false
```

运行时参数（通过 `Flash_fwd_params` 传递）:
- `is_causal` - ✅ 运行时决定（通过 CAUSAL_LOCAL_SWITCH 宏）
- `cu_seqlens_q/k` - ✅ 运行时决定（通过 VARLEN_SWITCH 宏）
- `q_descale_ptr` - ✅ 运行时决定（kernel 内部检查 nullptr）

---

## 编译和使用

### 编译

```bash
cd /home/qianxu/flash-attention/standalone
mkdir -p build && cd build
cmake ..
cmake --build . --target flash_attention_exec_v2
```

### 运行示例

```bash
# 运行所有三个测试用例
./build/flash_attention_exec_v2
```

---

## API 使用示例

### 最小示例 (FP16 Causal)

```cpp
#include "flash_api_v2.h"

int main() {
    // 分配 device 内存
    void *d_q, *d_k, *d_v, *d_out;
    cudaMalloc(&d_q, batch * seqlen * heads * headdim * sizeof(__half));
    // ... (K, V, out)

    // 设置参数
    flash::FlashAttentionParams params;
    params.q = d_q;
    params.k = d_k;
    params.v = d_v;
    params.out = d_out;
    params.batch_size = 2;
    params.seqlen_q = 512;
    params.seqlen_k = 512;
    params.num_heads = 16;
    params.num_heads_k = 16;
    params.head_dim = 128;
    params.is_causal = true;
    params.dtype = flash::DataType::FP16;

    // 运行
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    int result = flash::flash_attention_forward(params, stream);
    if (result != 0) {
        std::cerr << "Error: " << flash::get_error_string(result) << "\n";
    }
    cudaStreamSynchronize(stream);

    // 清理
    cudaFree(d_q); // ...
    cudaStreamDestroy(stream);
    return 0;
}
```

---

## 性能优化建议

1. **Decoding 场景**: 使用 GQA/MQA 减少 KV cache 大小
   ```cpp
   params.num_heads = 16;     // Q heads
   params.num_heads_k = 2;    // KV heads (8:1 ratio)
   ```

2. **Varlen 场景**: 按序列长度排序可以提高性能
   ```cpp
   // 将相似长度的序列放在同一个 batch
   // 例如: [512, 510, 508] 比 [512, 128, 32] 更高效
   ```

3. **FP8 场景**: 选择合适的 descale factor
   ```cpp
   // descale_factor 应该使 fp8_value * descale ≈ 原始 FP16 值
   // 太小: 精度损失
   // 太大: FP8 溢出
   float descale = max_abs_value / 448.0f;  // FP8 E4M3 max = 448
   ```

4. **Head dimension 选择**: 优先使用 128 (最优化)
   ```cpp
   // 性能排序: 128 > 64 > 256 > 96 > 192
   ```

---

## 错误处理

```cpp
int result = flash::flash_attention_forward(params, stream);

if (result != 0) {
    // 获取错误信息
    std::cerr << "Flash Attention failed: "
              << flash::get_error_string(result) << std::endl;

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
}
```

常见错误码:
- `-1`: Null pointer
- `-2`: Unsupported head_dim
- `-3`: FP8 head_dim constraint
- `-6`: Varlen cu_seqlens mismatch
- `-10` ~ `-16`: Unsupported features
- `-100`: CUDA runtime error

---

## 限制和注意事项

1. **只支持 Hopper (SM90a) GPU**
   - H100, H800
   - 不支持 Ampere (A100) 或更早的架构

2. **Varlen 模式的内存布局**
   ```
   正确: [total_tokens, heads, headdim]  # 所有序列连续
   错误: [batch, max_seqlen, heads, headdim]  # padding
   ```

3. **GQA 约束**
   ```cpp
   num_heads % num_heads_k == 0  // 必须整除
   num_heads_k <= num_heads      // KV heads 不能多于 Q heads
   ```

4. **FP8 输出格式**
   - 输入: FP8 E4M3
   - 输出: 总是 FP16/BF16 (不能是 FP8)

5. **Causal + Local 互斥**
   ```cpp
   // ❌ 错误
   params.is_causal = true;
   params.window_size_left = 256;  // 冲突!

   // ✅ 正确: 选择其一
   params.is_causal = true;
   params.window_size_left = -1;  // 或者
   params.is_causal = false;
   params.window_size_left = 256;
   ```

---

## 与完整版 Flash Attention 的区别

| 特性 | Standalone V2 | 完整版 (Python) |
|------|---------------|----------------|
| FP16/FP8 | ✅ | ✅ |
| BF16 | ❌ | ✅ |
| Causal mask | ✅ | ✅ |
| Varlen | ✅ | ✅ |
| GQA/MQA | ✅ | ✅ |
| Descaling | ✅ | ✅ |
| Softcapping | ❌ | ✅ |
| PackGQA | ❌ | ✅ |
| Split-KV | ❌ | ✅ |
| Paged KV | ❌ | ✅ |
| RoPE | ❌ | ✅ |
| Dropout | ❌ | ✅ |
| Backward pass | ❌ | ✅ |

---

## 参考资料

- [Flash Attention 3 论文](https://arxiv.org/abs/2407.08608)
- [CUTLASS 3.0 文档](https://github.com/NVIDIA/cutlass)
- [Hopper 架构白皮书](https://resources.nvidia.com/en-us-tensor-core)
