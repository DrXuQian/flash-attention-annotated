# 如何判断实例化的kernel是否一致

## 最底层、最准确的方法

### 方法1：在kernel launch点添加printf（最直接）

在 `hopper/flash_fwd_launch_template.h` 的 `run_flash_fwd()` 函数开头添加：

```cpp
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    // 只打印一次（每个template instantiation）
    static bool printed = false;
    if (!printed) {
        printed = true;
        printf("[KERNEL] Arch=%d Element_size=%zu hdim=%d/%d causal=%d varlen=%d\n",
               Arch, sizeof(Element), kHeadDim, kHeadDimV,
               (int)Is_causal, (int)Varlen);
        printf("  params: b=%d h=%d d=%d seqlen_q=%d total_q=%ld\n",
               params.b, params.h, params.d, params.seqlen_q, (long)params.total_q);
    }
    // ... rest of function
}
```

**优点：**
- 看到的是真正被调用的kernel template参数
- 可以看到runtime时的params值
- 不依赖任何外部工具

**缺点：**
- 需要修改hopper源码
- 需要重新编译

---

### 方法2：使用提供的自动化脚本

```bash
cd /home/qianxu/flash-attention/standalone

# 自动patch、编译、运行、对比
./scripts/compare_kernel_params.sh [pytorch_test.py]
```

这个脚本会：
1. 自动添加debug打印到hopper代码
2. 重新编译
3. 运行并提取kernel信息
4. 对比PyTorch和Standalone的差异
5. 自动恢复原始代码

---

### 方法3：手动添加打印到standalone

在 `standalone/src/flash_api.cu` 的 `dispatch_headdim()` 中：

```cpp
template<typename DType>
static int dispatch_headdim(
    Flash_fwd_params& flash_params,
    int head_dim,
    cudaStream_t stream
) {
    // 添加这个打印
    printf("[STANDALONE] Calling run_mha_fwd_<90, %s, %d, %d, false, false, false, false>\n",
           std::is_same<DType, cutlass::half_t>::value ? "FP16" : "FP8",
           head_dim, head_dim);
    printf("  Flash_fwd_params: b=%d h=%d seqlen_q=%d total_q=%ld cu_seqlens_q=%p\n",
           flash_params.b, flash_params.h, flash_params.seqlen_q,
           flash_params.total_q, flash_params.cu_seqlens_q);

    switch (head_dim) {
        case 128:
            run_mha_fwd_<90, DType, 128, 128, false, false, false, false>(
                flash_params, stream);
            return 0;
        // ...
    }
}
```

---

## 对比Checklist

当你添加了打印后，对比这些关键参数：

### Template参数（编译时）：
- [ ] `Arch` - 应该都是90
- [ ] `Element` - FP16/BF16/FP8
- [ ] `kHeadDim` - 64/96/128/192/256
- [ ] `kHeadDimV` - 通常等于kHeadDim
- [ ] `Is_causal` - true/false
- [ ] `Is_local` - false
- [ ] `Has_softcap` - false
- [ ] `Varlen` - true/false（由cu_seqlens是否为null决定）
- [ ] `PagedKVNonTMA` - false
- [ ] `AppendKV` - false
- [ ] `PackGQA` - false
- [ ] `Split` - false
- [ ] `V_colmajor` - false

### Runtime参数：
- [ ] `params.b` - batch size
- [ ] `params.h` - num_heads
- [ ] `params.h_k` - num_heads_k
- [ ] `params.d` - head_dim
- [ ] `params.seqlen_q` - **关键！**
- [ ] `params.seqlen_k` - **关键！**
- [ ] `params.total_q` - **关键！**
- [ ] `params.cu_seqlens_q` - **关键！NULL vs non-NULL**
- [ ] `is_varlen_q` (computed) - **关键！**
- [ ] `seqlen_q` (in run_flash_fwd) - **关键！**

### 计算出的常量（在run_flash_fwd内部）：
- [ ] `kBlockM` - tile大小
- [ ] `kBlockN` - tile大小
- [ ] `kStages` - pipeline stages
- [ ] `UsePersistentScheduler` - scheduler类型

---

## 如果发现不一致

### Template参数不一致：

说明调用的是不同的kernel variant。检查：
1. `dispatch_headdim()` 调用的template参数
2. 是否有条件编译（`#ifdef`）导致的差异

### Runtime参数不一致：

说明 `Flash_fwd_params` 设置有误。特别检查：
1. `cu_seqlens_q` - 必须严格为NULL（非varlen）或valid pointer（varlen）
2. `seqlen_q` vs `total_q` - varlen时应该设置max_seqlen_q
3. All strides - 必须完全一致

### 参数都一致但性能不同：

可能原因：
1. **编译选项不同** - 检查 `-O3`, `--use_fast_math`
2. **CUDA版本不同** - 检查 `nvcc --version`
3. **GPU状态不同** - 温度、频率
4. **测量方法不同** - event vs walltime

---

## 快速验证：使用cuda-memcheck

```bash
# Standalone
cuda-memcheck ./build/test_fp16_causal_gqa 2>&1 | grep "Kernel:"

# PyTorch (if possible)
cuda-memcheck python test.py 2>&1 | grep "Kernel:"
```

这会显示实际执行的kernel函数名，包含完整的template参数！

---

## 最终验证：符号表

```bash
# 查看编译后的kernel符号
cd standalone/build
cuobjdump -elf test_fp16_causal_gqa | grep "flash.*kernel"

# 或者
nm -C test_fp16_causal_gqa | grep "run_flash_fwd"
```

如果符号名称包含所有template参数，你可以直接对比。

---

## 示例输出

正确的输出应该类似：

```
========== KERNEL INSTANTIATION ==========
Template params: Arch=90 Element=FP16/BF16 kHeadDim=128 kHeadDimV=128
  Is_causal=1 Is_local=0 Has_softcap=0 Varlen=0
  PagedKVNonTMA=0 AppendKV=0 HasQv=0 PackGQA=0 Split=0 V_colmajor=0
==========================================

========== RUNTIME PARAMS ==========
  params.b=1 params.h=16 params.h_k=2 params.d=128
  params.seqlen_q=1285 params.seqlen_k=1285
  params.total_q=1285 params.total_k=1285
  is_varlen_q=0 is_varlen_k=0
  Computed: kBlockM=64 kBlockN=128 kStages=2
====================================
```

PyTorch和Standalone的输出应该**完全一致**！
