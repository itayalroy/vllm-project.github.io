---
layout: post
title: "Achieving near-zero-downtime scaling in Elastic EP"
author: "Itay Alroy (NVIDIA)"
summary: "How asynchronous preparation and NIXL EP enabled near-zero-downtime Elastic EP scaling in vLLM."
image: /assets/figures/2026-10-05-elastic-ep-near-zero-downtime/figure00-pytorch-slo.png
tags:
  - large-scale-serving
  - elastic-ep
  - expert-parallelism
  - moe
  - nixl
  - cuda-graphs
---

Before Elastic EP, changing a vLLM service's GPU count meant adding or removing whole model replicas, or restarting the engine with a different GPU count. As models grow larger, each replica needs more GPUs, so replica scaling adds or removes GPUs in increasingly large chunks. Restarting stopped our engine from serving for **197 seconds** (see our [benchmark](#performance-on-deepseek-v4-pro)).

In our [first Elastic EP post](https://vllm.ai/blog/2026-05-14-elastic-expert-parallelism), we showed how vLLM could add or remove GPUs from a running MoE engine without restarting it. But avoiding a restart did not mean keeping it serving. Traffic effectively stopped throughout reconfiguration while we initialized new workers, prepared the expanded communication groups, loaded model weights, and repeated kernel warmup and CUDA graph capture. In our scale-up benchmark, blocking preparation meant existing requests stopped progressing and new requests were rejected for **100 seconds** (full details in the [performance section](#performance-on-deepseek-v4-pro)).

In this blog, we describe how we brought Elastic EP reconfiguration to near-zero downtime. Asynchronous preparation lets existing workers keep serving while joining workers initialize, receive model weights, and prepare the target communication groups. But moving those tasks into the background was not enough: kernel warmup and CUDA graph capture still required existing workers to stop serving.

[NIXL EP](https://github.com/ai-dynamo/nixl/blob/main/examples/device/ep/README.md) helped remove the remaining warmup and graph-recapture bottleneck. It lets us expand the existing EP communicator while keeping existing workers' CUDA graphs valid, and lets joining workers warm up kernels and capture graphs independently while serving continues. We also modified vLLM to preserve graph-visible tensors and kernel configurations across reconfiguration. Once preparation finishes, only a short coordinated switch remains. On DeepSeek V4 Pro, we achieved **0.82 seconds of downtime for scale-up** and **1.96 seconds for scale-down**.

## Performance on DeepSeek V4 Pro

### Scale-up and scale-down timings

We benchmarked DeepSeek V4 Pro on GB200 GPUs, scaling from 16 to 20 GPUs and back. We compared three designs:

- **A: Blocking preparation.** All preparation, warmup, and graph capture happen while serving is paused.
- **B: Asynchronous preparation.** Startup, communication setup, and weight transfer run in the background; warmup and graph recapture still pause serving.
- **C: Asynchronous preparation + graph reuse.** Existing graphs stay valid, and joining workers prepare independently of serving workers. This design is supported only by NIXL EP.

**Downtime** is the observed serving interruption: in-flight requests stop progressing and new requests are not admitted. **Total time** is the duration of the full scaling operation, including preparation. The two durations are measured separately.

| Metric | A: Blocking | B: Async preparation | C: Async preparation + graph reuse |
| --- | ---: | ---: | ---: |
| 16 to 20 GPUs: downtime | 99.62 s | 58.04 s | **0.82 s** |
| 20 to 16 GPUs: downtime | 56.25 s | 44.07 s | **1.96 s** |
| 16 to 20 GPUs: total time | 99.72 s | 97.27 s | 92.34 s |
| 20 to 16 GPUs: total time | 56.21 s | 51.46 s | 9.32 s |

All configurations passed numerical correctness checks, including checks on every joining worker.

### Maintaining the SLO while adding GPUs

We served DeepSeek V4 Pro on 16 GPUs, then gradually increased the incoming request rate from about 8 to 16 requests per second. We replayed the same 11,391-request scenario against three deployments: one stayed at 16 GPUs, one restarted with 20 GPUs, and one scaled to 20 GPUs using Elastic EP with NIXL EP. The service-level objective (SLO) was a rolling p95 time to first token below 5 seconds.

![Elastic EP stays below the SLO throughout the scenario. During preparation, its serving performance matches the static 16-GPU deployment. After scaling, it matches the recovered 20-GPU deployment's performance.](/assets/figures/2026-10-05-elastic-ep-near-zero-downtime/figure00-pytorch-slo.png)

*Figure 1. End-to-end serving under rising request load. Curves show 30-second rolling p95 outcomes, with failed requests assigned a 30-second penalty rather than a measured TTFT.*

- Elastic EP completed all 11,391 requests and stayed below the SLO throughout the scenario.
- During preparation, Elastic EP matched the static 16-GPU deployment's serving performance, with no visible degradation from background preparation.
- After scaling, Elastic EP matched the recovered 20-GPU deployment's performance, with no visible post-scale degradation.

## Prepare the target topology while the current one serves

We reorganized vLLM's reconfiguration flow into background preparation and a short commit. Each existing worker prepares its part of the new topology on a separate thread while continuing to serve requests with its current communication groups and CUDA graphs. In parallel, joining workers initialize, receive model weights, and prepare their communication and GPU execution. At commit, vLLM briefly pauses its schedulers, activates the prepared topology, and resumes serving. These changes were introduced in [asynchronous preparation](https://github.com/vllm-project/vllm/pull/47288) and [prepare/commit optimizations](https://github.com/vllm-project/vllm/pull/51885).

Preparation covers four main tasks:

1. **Initialize joining workers.** Start their processes and allocate model and execution storage.
2. **Prepare communication.** Set up the target communication groups and expand the EP communicator.
3. **Transfer model weights.** Move the weights needed by the joining workers from existing workers over NVLink or RDMA.
4. **Prepare execution.** Complete the required compilation, kernel warmup, tuning setup, and CUDA graph construction on joining workers.

After activation, asynchronous expert parallel load balancing (EPLB) lets new workers start handling attention while existing workers handle expert computation. EPLB then redistributes experts across the expanded group in the background.

The difficult part is preparing communication and GPU execution without involving or disrupting serving workers.

## Expand EP communication in place

Fixed-membership EP communication libraries (the EP communication backends available in vLLM before NIXL EP) create an instance for a particular rank set. Adding or removing ranks requires a new instance. Even if it can be created in the background, there are still two problems.

First, independent old and new instances need separate communication buffers and resources such as memory registrations, IPC handles, and RDMA queue pairs (QPs), among others. Keeping both alive increases communication memory usage and repeats setup rather than reusing existing resources.

Second, switching to another EP communicator changes the buffers and communication state used by the EP communication kernels. Previously captured CUDA graphs are therefore no longer valid and must be recaptured before serving can resume. That requires repeating warmup and graph construction for many batch shapes across all retained workers. In our DeepSeek V4 Pro benchmark, even with asynchronous preparation, keeping warmup and graph recapture in commit meant serving pauses of **58 seconds during scale-up** and **44 seconds during scale-down**.

To solve these problems, we designed **NIXL EP**, an EP communication library tailor-made for Elastic EP. It adds or removes ranks in place, reusing the same NIXL agent, buffers, and retained-peer connections. New peers can be connected while existing workers keep serving. Only joining peers need new connections and registrations. It also keeps captured EP dispatch and combine entries valid across reconfiguration, enabling retained workers to reuse their existing CUDA graphs.

During preparation, vLLM's background thread connects joining peers without activating them. Commit then activates the prepared peers without replacing the communicator.

![The same NIXL EP communicator progresses from the current active group to connected inactive joining peers, then activates the target group at commit.](/assets/figures/2026-10-05-elastic-ep-near-zero-downtime/figure02-nixl-ep-lifecycle.png)

*Figure 2. Connecting and activating new EP peers. Green ranks are active; gray ranks are connected but masked. At commit, new peers are activated without replacing the agent or buffers.*

vLLM still creates new DP coordination groups and EPLB groups for the target topology, **but their communication runs outside the captured model forward**, so switching to them does not invalidate CUDA graphs.

## Preserve CUDA graphs on existing workers

A CUDA graph records GPU operations so that vLLM can replay a forward pass without repeatedly launching each operation from the CPU. A captured kernel node includes its launch configuration and argument values, including input and output tensor pointers. Memory contents can change between replays, but the recorded pointers must remain valid. These are part of the [CUDA graph execution contract](https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs).

vLLM [captures separate graphs for different batch shapes and execution modes](https://docs.vllm.ai/en/latest/design/cuda_graphs/), with warmup forwards before each graph capture.

Preserving captured CUDA graphs requires more than keeping the communicator object alive. NIXL EP also preserves the kernel launch configurations, graph-visible communication arguments, and input and output tensor addresses across reconfiguration. Its kernels read active group membership from device state that can be updated in place.

We also changed vLLM to keep the surrounding compute kernels' captured arguments, launch configurations, and input and output tensor addresses stable across reconfiguration:

- We reuse retained workers' fused-MoE execution objects instead of rebuilding them for a different active EP size.
- We keep graph-visible workspace and dispatch output addresses and layouts stable, so captured expert kernels can continue using them.
- We preserve expert-routing tensors' backing storage and update mappings in place. EPLB can change the mapping contents without moving the tensors used by recorded kernels.

Dispatch outputs are also inputs to expert computation. If their addresses or layouts changed with EP size, the captured expert kernels would still use the old pointers and offsets. Both NIXL EP and vLLM preserve these addresses and layouts across reconfiguration.

Together, these changes allow retained workers to reuse their existing graphs after both scale-up and scale-down.

## Prepare joining workers independently

Existing workers have graphs to preserve. Joining workers have none.

Joining workers' preparation normally runs model forwards with EP dispatch and combine, along with exchanges of batch information between workers. These steps require other workers to participate, but existing workers are already serving real requests on the current topology.

NIXL EP lets each joining worker temporarily mask every remote EP peer during warmup and graph preparation. Only that worker's local rank is active for EP communication. We also modified vLLM so these preparation forwards do not wait for DP batch information from other workers. This lets joining workers run their dummy batches and capture CUDA graphs without participation from existing workers.

![Existing workers serve with their current peers active and joining peers masked. Each joining worker keeps only itself active for local warmup and graph capture, masking all remote peers, including other joining workers.](/assets/figures/2026-10-05-elastic-ep-near-zero-downtime/figure04-independent-capture.png)

*Figure 3. Green ranks are active for EP communication; gray ranks are masked. Existing workers serve together while each joining worker masks all remote EP peers for preparation.*

After capture, NIXL EP can reactivate peers without invalidating the graphs. Joining workers reuse their locally captured graphs when serving begins on the expanded group.

## Padding in expert-kernel inputs

To keep CUDA graphs stable across reconfiguration, we size dispatch outputs (the inputs to expert kernels) for the maximum EP capacity. Expert inputs are padded even in a static deployment, but reserving maximum capacity increases the padding. Backends that do not efficiently skip padded token slots incur more overhead as a result.

In a separate capacity benchmark, we kept the active EP group at 16 workers and compared a maximum capacity of 64 against 16. These measurements use DeepSeek-V3 on 16 GB200 GPUs, with backend-specific weight formats.

| Expert backend | Decode throughput change | Prefill throughput change |
| --- | ---: | ---: |
| Batched DeepGEMM | 0.08% higher | 0.97% lower |
| FlashInfer CuteDSL | 1.21% higher | 0.89% lower |
| Batched Triton | 18.31% lower | 19.21% lower |
| Batched CUTLASS FP8 | 77.49% lower | 63.90% lower |

DeepGEMM and CuteDSL throughput is effectively unaffected in this comparison. Triton and CUTLASS still process parts of the padded buffers, so their padding overhead grows with the configured maximum.

We also plan to reduce NIXL EP's extra memory cost with [CUDA virtual memory management (VMM)](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html): reserve stable virtual addresses, but back only the regions needed by the active group with physical HBM. This would let buffers grow without changing the addresses recorded in CUDA graphs or allocating all maximum-capacity HBM up front.

## Try Elastic EP

Background preparation, CUDA graph reuse, and independent graph capture on joining workers are all available today in the latest [vLLM](https://pypi.org/project/vllm/) and [NIXL](https://pypi.org/project/nixl/) releases. Install them with pip and start using Elastic EP.

For a complete usage example, see the [`cuda_graphs_heavy_nixl_ep` case in vLLM's Elastic EP test](https://github.com/vllm-project/vllm/blob/v0.30.0/tests/distributed/test_elastic_ep.py#L224). It includes the server configuration and scaling API calls, scales from two GPUs to four and back while sending requests, and checks serving during preparation and model accuracy around each resize.

## What comes next

### Restore warm workers with Dynamo Snapshot

We have effectively solved the long serving interruption. The next challenge is getting new capacity ready sooner. We plan to integrate [Dynamo Snapshot](https://developer.nvidia.com/blog/nvidia-dynamo-snapshot-fast-startup-for-inference-workloads-on-kubernetes/) so joining workers can restore initialized process and CUDA state instead of repeating compilation, kernel warmup, graph construction, and other startup work.

Communication for the target topology is not restored from the snapshot and must be prepared separately. Restoring large model weights from checkpoint storage can dominate restore time, so we plan to keep Elastic EP's GPU-to-GPU weight transfer from existing workers instead.

### Achieve state-of-the-art EP communication with NIXL EP

We aim for state-of-the-art dispatch and combine performance with NIXL EP while maintaining full support for Elastic EP and fault tolerance. An upcoming NIXL EP performance update is the next step toward that goal.

### Connect Elastic EP to Dynamo and llm-d orchestration

With the engine-level capability in place, the next step is to use Elastic EP from orchestrator autoscalers. Work toward this integration is already underway across the vLLM, Dynamo, and llm-d communities.

These efforts include [vLLM's external-load-balancer work](https://github.com/vllm-project/vllm/pull/43202) and [Dynamo's Elastic EP proposal](https://github.com/ai-dynamo/dynamo/issues/13121).

If you run inference at scale, we encourage you to try Elastic EP and share your workloads and operational requirements. If you build inference infrastructure, we welcome contributions to vLLM Elastic EP and NIXL EP. Let's get Elastic EP running in production.

## Acknowledgements

- Tyler Michael Smith and Sage Moore from Red Hat for their invaluable feedback and thorough reviews throughout this work.
- Linoy Geva (NVIDIA) for her work on serving performance during preparation.
- The entire NIXL team for their efforts on NIXL EP.
- Tzu-Ling Kan, Julien Mancuso, and the entire Dynamo team for their efforts to support Elastic EP in Dynamo.
