# Debug Guide: 为什么参数一致但结果不同？

如果 `Flash_fwd_params` 的所有字段都打印一致，但执行时间和instructions不同，可能的原因：

## 1. Kernel Template参数不同

即使 `flash_params` 一致，`run_mha_fwd_<...>` 的模板参数可能不同。

### 检查方法：

**在PyTorch hopper/flash_api.cpp中添加打印：**

```cpp
// 在 mha_fwd() 函数中，调用 run_mha_fwd_ 之前
printf("[PYTORCH] run_mha_fwd_ template params:\n");
printf("  Arch: %d\n", Arch);
printf("  Element: %s\n", typeid(Element).name());
printf("  kHeadDim: %d\n", kHeadDim);
printf("  kHeadDimV: %d\n", kHeadDimV);
printf("  Split: %d\n", Split);
printf("  PagedKVNonTMA: %d\n", PagedKVNonTMA);
printf("  Has_softcap: %d\n", Has_softcap);
printf("  PackGQA: %d\n", PackGQA);
```

**在Standalone src/flash_api.cu中添加打印：**

已经在 `dispatch_headdim()` 中添加了debug参数，使用：
```cpp
dispatch_headdim<cutlass::half_t>(flash_params, params.head_dim, stream, true);
```

### 对比：
- Arch应该都是90
- Element: PyTorch可能是 `cutlass::half_t` 或 `at::Half`
- Split, PagedKVNonTMA, Has_softcap, PackGQA 应该都是false

---

## 2. CUDA编译选项不同

### 检查CMakeLists.txt：

**PyTorch版本编译选项（flash-attention/hopper/CMakeLists.txt）：**
```cmake
-gencode arch=compute_90a,code=sm_90a
-O3
--use_fast_math
-DNDEBUG
```

**Standalone版本编译选项（standalone/CMakeLists.txt）：**
```bash
cd standalone/build
cmake .. -DCMAKE_BUILD_TYPE=Release
```

检查是否有：
- 相同的 `-gencode` flags
- 相同的优化级别 (-O3)
- 相同的 `--use_fast_math`

### 对比命令：
```bash
# 看PyTorch编译命令
cd flash-attention/hopper
make VERBOSE=1

# 看Standalone编译命令
cd standalone/build
make VERBOSE=1
```

---

## 3. 使用CUDA Profiler对比实际kernel

这是最准确的方法！

### 步骤：

**1. Profile PyTorch版本：**
```bash
cd flash-attention
nsys profile -o pytorch_profile python -c "
import torch
from flash_attn import flash_attn_func

q = torch.randn(1, 1285, 16, 128, dtype=torch.float16, device='cuda')
k = torch.randn(1, 1285, 16, 128, dtype=torch.float16, device='cuda')
v = torch.randn(1, 1285, 16, 128, dtype=torch::float16, device='cuda')

out = flash_attn_func(q, k, v, causal=True)
torch.cuda.synchronize()
"
```

**2. Profile Standalone版本：**
```bash
cd standalone/build
nsys profile -o standalone_profile ./test_fp16_causal_gqa
```

**3. 对比kernel：**
```bash
cd standalone
python scripts/compare_kernels.py \
    ../pytorch_profile.nsys-rep \
    build/standalone_profile.nsys-rep
```

### 分析结果：

如果有不同的kernel被调用：
- **PyTorch多了某些kernel** → 可能是额外的优化路径
- **Standalone多了某些kernel** → 可能是错误的code path
- **相同kernel但调用次数不同** → 逻辑错误
- **相同kernel但时间不同** → 编译优化差异

---

## 4. 检查运行时分支选择

即使参数一致，kernel内部可能走不同的分支。

### 添加printf debug：

**在hopper/flash_fwd_launch_template.h中添加：**

```cpp
void run_mha_fwd_(...) {
    // 在函数开头
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("[KERNEL] use_prepare_varlen: %d\n", use_prepare_varlen);
        printf("[KERNEL] is_varlen_q: %d\n", is_varlen_q);
        printf("[KERNEL] seqlen_q: %d\n", seqlen_q);
        printf("[KERNEL] batch_q: %d\n", batch_q);
    }
    // ...
}
```

---

## 5. 检查具体的差异

### Instructions数量不同：

可能原因：
- **Loop unrolling不同** - 编译器优化
- **Register spilling不同** - 寄存器压力
- **分支预测不同** - if/else选择

### 时间不同：

可能原因：
- **Memory access pattern不同** - 缓存命中率
- **Occupancy不同** - 并行度
- **Warp divergence不同** - 分支效率

### 使用nsight compute深度分析：

```bash
# Profile单个kernel
ncu --set full -o pytorch_kernel python test.py
ncu --set full -o standalone_kernel ./test_standalone

# 对比metrics
ncu --import pytorch_kernel.ncu-rep standalone_kernel.ncu-rep
```

关注：
- `inst_executed`: 执行的指令数
- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`: 全局内存读取
- `sm__warps_active.avg`: 平均活跃warp数
- `smsp__sass_thread_inst_executed_op_*.sum`: 不同操作的指令数

---

## 6. 最简单的检查：编译选项

**快速验证是否是编译问题：**

```bash
cd standalone/build
rm -rf *
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CUDA_FLAGS="-O3 --use_fast_math -lineinfo"
make VERBOSE=1 | grep "nvcc.*flash_fwd"
```

检查nvcc命令是否包含：
- `-O3`
- `--use_fast_math`
- `-gencode arch=compute_90a,code=sm_90a`

---

## 7. 最终验证：数值结果

虽然时间不同，但数值结果应该一致（在误差范围内）。

```cpp
// 在test中添加
float* h_out_pytorch = ...;
float* h_out_standalone = ...;

float max_diff = 0.0f;
for (int i = 0; i < total_elements; i++) {
    float diff = fabs(h_out_pytorch[i] - h_out_standalone[i]);
    max_diff = max(max_diff, diff);
}

printf("Max difference: %e\n", max_diff);
// FP16: 应该 < 1e-3
// FP8:  应该 < 1e-2
```

如果数值结果一致，那么性能差异可能是：
- 正常的编译优化差异
- GPU状态不同（温度、频率）
- 测量方法不同

---

## 总结：Debug优先级

1. ✅ 先用nsys对比kernel - 最直接
2. ✅ 检查CMakeLists编译选项 - 最常见原因
3. ✅ 对比template参数 - 确保调用同一个kernel
4. ❓ 深度profiling (ncu) - 如果上述都没问题
5. ❓ 修改kernel源码加printf - 最后手段

如果所有参数、template、编译选项都一致，但仍然有差异，那可能是：
- PyTorch的额外wrapper开销
- 不同的stream synchronization策略
- GPU driver版本差异
