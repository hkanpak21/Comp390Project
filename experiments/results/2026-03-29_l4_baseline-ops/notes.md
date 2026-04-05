# Run Notes: 2026-03-29 `l4` `baseline-ops`

## What We Ran
- `ckks_bench`: Phantom's built-in CKKS benchmark across N=8192, 16384, 32768, 65536
- `nccl_bandwidth_test --n-gpus 1 --msg-size-mb 21 --iters 50`: NCCL loopback on single L4

## What We Expected
- Multiply+relin at N=65536: ~10-30 ms range based on A100 (37s / ~500 operations)
- NCCL loopback: limited by GPU memory bandwidth, not network (~300 GB/s on L4)

## What We Got

### Key numbers at N=65536, L=20 (NEXUS GELU params):
| Operation           | Median (µs) |
|---------------------|-------------|
| multiply            | ~30,194     |
| rotate_vector       | ~29,482     |
| rescale_to_next     | ~1,426      |
| add                 | ~566        |
| multiply_plain      | ~566        |
| gen_relinkey        | ~54,560     |

### NCCL (single-GPU loopback):
- AllGather 21 MB: **21 µs**
- This confirms: on real NVSwitch (600 GB/s), AllGather << compute time

## Anomalies / Surprises
- multiply and rotate are both ~30 ms at N=65536,L=20. This makes sense since rotation = key-switch = same cost as multiply+relin.
- L4 is ~5-10x slower than A100 at FHE (A100 does BERT in 37s ≈ ~500 ops × ~74ms avg, but many ops are cheaper than multiply).
- NEXUS benchmarks (bert_inference, bootstrapping_bench) could not be compiled because:
  1. NEXUS's `ckks_evaluator.cuh` expects a specific older Phantom API (global free functions like `::multiply_inplace`)
  2. Our `vendor/phantom` is a newer version with a different API
  3. NEXUS's own bundled phantom (`vendor/nexus/thirdparty/phantom-fhe`) was not included in the rsync
  - **Fix for next session**: either init the NEXUS phantom-fhe submodule, or build NEXUS with its own cmake (needs CUDA 11 path override)
- `nccl_bandwidth_test` compiled and ran successfully with our multi-GPU library.

## Next Steps
- **Immediate**: Stop instance, resubmit AWS quota request for p4d.24xlarge citing billing history
- **Next EC2 session**: Build NEXUS with its own cmake (override CUDA path to 13.0), run full BERT-base baseline
- **Alternative**: Use Phantom's CKKS benchmark numbers to extrapolate BERT-base latency on L4
