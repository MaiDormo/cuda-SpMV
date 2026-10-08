# NVIDIA H100 GPU Specifications (reference for sm_90 fatbin)

Reference values for the Hopper H100 SXM/PCIe variant (check `nvidia-smi`
on the actual node; values vary by SKU). Included so the multi-arch
fatbin (`sm_80;sm_89;sm_90` + `compute_90` PTX) has a roofline reference
next to `gpu_info.md` (A30) and `l40s_info.md` (L40S).

## Basic Information
| Specification | Value |
|---------------|-------|
| Device | NVIDIA H100 (Hopper) |
| CUDA Capability | 9.0 |
| Total Global Memory | ~80 GB HBM3 (SXM) / ~96 GB HBM3 (H200) |

## Core Architecture
| Specification | Value |
|---------------|-------|
| Multiprocessors | 144 (SXM) / 132 (PCIe) |
| GPU Max Clock Rate | ~1980 MHz (boost) |

## Memory Specifications
| Specification | Value |
|---------------|-------|
| Memory Bandwidth | ~3 TB/s (H100 SXM) / ~4.8 TB/s (H200) |
| L2 Cache Size | ~50 MB |
| Shared Memory per Block | up to ~228 KB (per SM configurable) |
| Registers per SM | 65536 |

## Notes for this repo
- Our kernels use only portable features (`__shfl_xor_sync`, `__ldg`,
  `__ldcs`); no WGMMA/TMA, so the `sm_90` cubin needs no code change.
- `L2_FLUSH_BYTES` is 128 MB in `src/spmv_gpu_hybrid_v2_csr.cu`, which covers
  the largest L2 in the fleet (L40S 96 MB). H100 (50 MB) is covered too.
- Benchmark scripts log `nvidia-smi --query-gpu=name,compute_cap,...`
  (`GPU arch probe`) so results can be grouped by arch in the CSV.
