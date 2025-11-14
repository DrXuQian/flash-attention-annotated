#include "../include/flash_api.h"
#include <iostream>
#include <vector>
#include <random>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

void print_usage(const char* program_name) {
    std::cout << "Flash Attention Standalone - Test Program\n";
    std::cout << "=========================================\n\n";
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -h, --help              Show this help message\n";
    std::cout << "  -d, --dtype TYPE        Data type: fp16, fp8 (default: fp16)\n";
    std::cout << "  -b, --batch SIZE        Batch size (default: 1)\n";
    std::cout << "  -q, --seqlen-q LENGTH   Query sequence length (default: 512)\n";
    std::cout << "  -k, --seqlen-k LENGTH   Key/Value sequence length (default: 512)\n";
    std::cout << "  -n, --num-heads HEADS   Number of query attention heads (default: 16)\n";
    std::cout << "  -v, --num-kv-heads KV   Number of key-value heads for GQA (default: 16)\n";
    std::cout << "                          Use fewer than num-heads for GQA (e.g., 2 for Qwen2.5-VL)\n";
    std::cout << "  -m, --head-dim DIM      Head dimension: 64, 96, 128, 192, 256 (default: 128)\n";
    std::cout << "  -c, --causal            Enable causal masking (default: false)\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << "                      # Run with defaults (MHA: 16 heads)\n";
    std::cout << "  " << program_name << " -d fp8              # Run with FP8 E4M3\n";
    std::cout << "  " << program_name << " -n 16 -v 2          # GQA: 16 Q heads, 2 KV heads (Qwen2.5-VL)\n";
    std::cout << "  " << program_name << " -b 2 -q 1024        # Batch=2, seqlen=1024\n";
    std::cout << "  " << program_name << " -c                  # Enable causal attention\n\n";
    std::cout << "Supported configurations:\n";
    std::cout << "  Data types: FP16, FP8_E4M3\n";
    std::cout << "  Head dimensions: 128, 256\n";
    std::cout << "  Attention: MHA (num-heads == num-kv-heads) or GQA (num-kv-heads < num-heads)\n";
    std::cout << "  Architecture: Hopper (SM90a)\n";
}

// Helper function to generate random FP16 data
void generate_random_fp16(std::vector<__half>& data) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = __float2half(dis(gen));
    }
}

// Helper function to convert FP32 to FP8 E4M3
void generate_random_fp8_e4m3(std::vector<__nv_fp8_e4m3>& data) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = __nv_fp8_e4m3(dis(gen));
    }
}

int main(int argc, char** argv) {
    // Default configuration for Qwen2.5-VL-3B with GQA
    flash::DataType dtype = flash::DataType::FP16;
    int batch_size = 1;
    int seqlen_q = 512;
    int seqlen_k = 512;
    int num_heads = 16;       // Number of query heads
    int num_kv_heads = 16;    // Number of KV heads (16=MHA, 2=GQA for Qwen2.5-VL)
    int head_dim = 128;
    bool is_causal = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        else if (arg == "-d" || arg == "--dtype") {
            if (i + 1 < argc) {
                std::string dtype_str = argv[++i];
                if (dtype_str == "fp16" || dtype_str == "FP16") {
                    dtype = flash::DataType::FP16;
                } else if (dtype_str == "fp8" || dtype_str == "FP8") {
                    dtype = flash::DataType::FP8_E4M3;
                } else {
                    std::cerr << "Error: Invalid dtype '" << dtype_str << "'. Use fp16 or fp8.\n";
                    return 1;
                }
            } else {
                std::cerr << "Error: --dtype requires an argument\n";
                return 1;
            }
        }
        else if (arg == "-b" || arg == "--batch") {
            if (i + 1 < argc) {
                batch_size = std::atoi(argv[++i]);
                if (batch_size <= 0) {
                    std::cerr << "Error: batch size must be positive\n";
                    return 1;
                }
            }
        }
        else if (arg == "-q" || arg == "--seqlen-q") {
            if (i + 1 < argc) {
                seqlen_q = std::atoi(argv[++i]);
                if (seqlen_q <= 0) {
                    std::cerr << "Error: seqlen_q must be positive\n";
                    return 1;
                }
            }
        }
        else if (arg == "-k" || arg == "--seqlen-k") {
            if (i + 1 < argc) {
                seqlen_k = std::atoi(argv[++i]);
                if (seqlen_k <= 0) {
                    std::cerr << "Error: seqlen_k must be positive\n";
                    return 1;
                }
            }
        }
        else if (arg == "-n" || arg == "--num-heads") {
            if (i + 1 < argc) {
                num_heads = std::atoi(argv[++i]);
                if (num_heads <= 0) {
                    std::cerr << "Error: num_heads must be positive\n";
                    return 1;
                }
            }
        }
        else if (arg == "-v" || arg == "--num-kv-heads") {
            if (i + 1 < argc) {
                num_kv_heads = std::atoi(argv[++i]);
                if (num_kv_heads <= 0) {
                    std::cerr << "Error: num_kv_heads must be positive\n";
                    return 1;
                }
            }
        }
        else if (arg == "-m" || arg == "--head-dim") {
            if (i + 1 < argc) {
                head_dim = std::atoi(argv[++i]);
                if (head_dim != 64 && head_dim != 96 && head_dim != 128 &&
                    head_dim != 192 && head_dim != 256) {
                    std::cerr << "Error: head_dim must be 64, 96, 128, 192, or 256\n";
                    return 1;
                }
            }
        }
        else if (arg == "-c" || arg == "--causal") {
            is_causal = true;
        }
        else {
            std::cerr << "Error: Unknown option '" << arg << "'\n";
            std::cerr << "Use -h or --help for usage information\n";
            return 1;
        }
    }

    // Validate FP8 constraints
    if (dtype == flash::DataType::FP8_E4M3 && head_dim != 128) {
        std::cerr << "Error: FP8 E4M3 only supports head_dim=128\n";
        return 1;
    }

    // Validate GQA configuration
    if (num_kv_heads > num_heads) {
        std::cerr << "Error: num_kv_heads (" << num_kv_heads
                  << ") cannot be greater than num_heads (" << num_heads << ")\n";
        return 1;
    }
    if (num_heads % num_kv_heads != 0) {
        std::cerr << "Error: num_heads (" << num_heads
                  << ") must be divisible by num_kv_heads (" << num_kv_heads << ")\n";
        return 1;
    }

    const char* dtype_name = (dtype == flash::DataType::FP16) ? "FP16" : "FP8_E4M3";
    const char* attn_type = (num_heads == num_kv_heads) ? "MHA" :
                           (num_kv_heads == 1) ? "MQA" : "GQA";

    std::cout << "Flash Attention Standalone Test" << std::endl;
    std::cout << "================================" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Data type: " << dtype_name << std::endl;
    std::cout << "  Attention type: " << attn_type;
    if (num_kv_heads != num_heads) {
        std::cout << " (ratio " << (num_heads / num_kv_heads) << ":1)";
    }
    std::cout << std::endl;
    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Sequence length Q: " << seqlen_q << std::endl;
    std::cout << "  Sequence length K/V: " << seqlen_k << std::endl;
    std::cout << "  Query heads: " << num_heads << std::endl;
    std::cout << "  KV heads: " << num_kv_heads << std::endl;
    std::cout << "  Head dimension: " << head_dim << std::endl;
    std::cout << "  Hidden size: " << (num_heads * head_dim) << std::endl;
    std::cout << "  Causal masking: " << (is_causal ? "enabled" : "disabled") << std::endl;
    std::cout << std::endl;

    // Calculate tensor sizes based on dtype
    // Q: [batch, seqlen_q, num_heads, head_dim]
    // K,V: [batch, seqlen_k, num_kv_heads, head_dim] for GQA
    size_t q_elements = batch_size * seqlen_q * num_heads * head_dim;
    size_t kv_elements = batch_size * seqlen_k * num_kv_heads * head_dim;
    size_t element_size = (dtype == flash::DataType::FP16) ? sizeof(__half) : sizeof(__nv_fp8_e4m3);
    size_t q_bytes = q_elements * element_size;
    size_t kv_bytes = kv_elements * element_size;

    std::cout << "Generating random test data..." << std::endl;
    std::cout << "  Q elements: " << q_elements << " (" << (q_bytes / 1024.0 / 1024.0) << " MB)" << std::endl;
    std::cout << "  K/V elements: " << kv_elements << " each (" << (kv_bytes / 1024.0 / 1024.0) << " MB each)" << std::endl;

    // Allocate device memory
    void *d_q, *d_k, *d_v, *d_out;
    float *d_softmax_lse;

    cudaMalloc(&d_q, q_bytes);
    cudaMalloc(&d_k, kv_bytes);
    cudaMalloc(&d_v, kv_bytes);
    cudaMalloc(&d_out, q_bytes);  // Output has same shape as Q
    cudaMalloc(&d_softmax_lse, batch_size * num_heads * seqlen_q * sizeof(float));

    // Generate random test data on host and copy to device
    if (dtype == flash::DataType::FP16) {
        std::vector<__half> h_q(q_elements);
        std::vector<__half> h_k(kv_elements);
        std::vector<__half> h_v(kv_elements);

        std::cout << "  Generating FP16 data..." << std::endl;
        generate_random_fp16(h_q);
        generate_random_fp16(h_k);
        generate_random_fp16(h_v);

        cudaMemcpy(d_q, h_q.data(), q_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_k, h_k.data(), kv_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_v, h_v.data(), kv_bytes, cudaMemcpyHostToDevice);
    } else {
        std::vector<__nv_fp8_e4m3> h_q(q_elements);
        std::vector<__nv_fp8_e4m3> h_k(kv_elements);
        std::vector<__nv_fp8_e4m3> h_v(kv_elements);

        std::cout << "  Generating FP8 E4M3 data..." << std::endl;
        generate_random_fp8_e4m3(h_q);
        generate_random_fp8_e4m3(h_k);
        generate_random_fp8_e4m3(h_v);

        cudaMemcpy(d_q, h_q.data(), q_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_k, h_k.data(), kv_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_v, h_v.data(), kv_bytes, cudaMemcpyHostToDevice);
    }
    std::cout << "  ✓ Test data generated and copied to device" << std::endl;
    std::cout << std::endl;

    // Set up Flash Attention parameters
    flash::FlashAttentionParams params;
    params.q = d_q;
    params.k = d_k;
    params.v = d_v;
    params.out = d_out;
    params.softmax_lse = d_softmax_lse;

    params.batch_size = batch_size;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.num_heads = num_heads;
    params.num_heads_k = num_kv_heads;  // Support GQA/MQA/MHA
    params.head_dim = head_dim;

    params.is_causal = is_causal;
    params.dtype = dtype;

    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Run Flash Attention
    std::cout << "Running Flash Attention forward pass..." << std::endl;
    int result = flash::flash_attention_forward(params, stream);

    if (result == 0) {
        std::cout << "✓ Flash Attention completed successfully!" << std::endl;
    } else {
        std::cout << "✗ Flash Attention failed with error: "
                  << flash::get_error_string(result) << std::endl;
    }

    // Synchronize and check for errors
    cudaStreamSynchronize(stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    // Cleanup
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_out);
    cudaFree(d_softmax_lse);
    cudaStreamDestroy(stream);

    return result;
}
