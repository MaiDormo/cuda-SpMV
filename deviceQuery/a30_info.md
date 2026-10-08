# NVIDIA A30 GPU Specifications

## Basic Information
| Specification | Value |
|---------------|-------|
| Device | NVIDIA A30 |
| CUDA Driver / Runtime Version | 12.5 / 12.1 |
| CUDA Capability Major/Minor | 8.0 |
| Total Global Memory | 24169 MB (25,343,295,488 bytes) |

## Core Architecture
| Specification | Value |
|---------------|-------|
| Multiprocessors | 56 |
| CUDA Cores per MP | 64 |
| Total CUDA Cores | 3584 |
| GPU Max Clock Rate | 1440 MHz (1.44 GHz) |

## Memory Specifications
| Specification | Value |
|---------------|-------|
| Memory Clock Rate | 1215 MHz |
| Memory Bus Width | 3072-bit |
| L2 Cache Size | 25,165,824 bytes (~24 MB) |
| Constant Memory | 65,536 bytes (64 KB) |
| Shared Memory per Block | 49,152 bytes (48 KB) |
| Shared Memory per MP | 167,936 bytes (~164 KB) |

## Thread & Block Limits
| Specification | Value |
|---------------|-------|
| Warp Size | 32 |
| Max Threads per MP | 2048 |
| Max Threads per Block | 1024 |
| Max Thread Block Dimensions | (1024, 1024, 64) |
| Max Grid Dimensions | (2147483647, 65535, 65535) |

## Texture & Memory Features
| Specification | Value |
|---------------|-------|
| Max Texture Dimensions 1D | 131072 |
| Max Texture Dimensions 2D | (131072, 65536) |
| Max Texture Dimensions 3D | (16384, 16384, 16384) |
| Max Layered 1D Texture | 1D=(32768), 2048 layers |
| Max Layered 2D Texture | 2D=(32768, 32768), 2048 layers |
| Maximum Memory Pitch | 2147483647 bytes |
| Texture Alignment | 512 bytes |
| Registers Available per Block | 65536 |

## Advanced Features
| Feature | Support |
|---------|---------|
| Concurrent Copy & Kernel Execution | Yes (3 copy engines) |
| Kernel Runtime Limit | No |
| Host Memory Sharing | No |
| Host Page-locked Memory Mapping | Yes |
| Surface Alignment Requirement | Yes |
| ECC Support | Enabled |
| Unified Addressing (UVA) | Yes |
| Managed Memory | Yes |
| Compute Preemption | Yes |
| Cooperative Kernel Launch | Yes |
| MultiDevice Co-op Kernel Launch | Yes |

## Metrics Asked For Lab
| Metric | Value |
|--------|-------|
| Operations per Second | 2430 MHz (Memory clock rate × 2 due to DDR) |
| Total Number of Byte Exchanged | 384 Byte (Memory bus width / 8 (bit in a Byte)) |
| Streaming Multiprocessor | 56 (as Referenced Before) |