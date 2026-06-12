# Profile-Aware AI on LUMI

A hands-on tutorial for understanding what your GPU is actually doing during an AI job on LUMI-G.

> What is your GPU actually doing?

This repository helps AI and HPC users answer that question with evidence. The goal is not to instrument every job. The goal is to pick the right level of detail for the question you are asking: run a light profile first, collect evidence about GPU utilization and memory, and only go deeper when that evidence points to a specific bottleneck.

Use this repo when you want to:

- check whether a GPU is actually being kept busy during a job
- detect per-GPU imbalance across GCDs on a single node
- identify which kernels dominate execution time
- trace host–GPU interaction and find launch gaps or memory-copy stalls
- read profiling output and decide the next action

> [!IMPORTANT]
> Profiling is a decision, not overhead. Pick the depth that matches the question you are asking. Start with light mode. Only go deeper when the current level gives you a reason to.

## The Profiling Ladder

Three tools, three questions:

| Level | Tool | Question answered | Runtime cost |
|---|---|---|---|
| Light | `rocm-smi` sidecar | Is my GPU busy? | Negligible |
| Deep-trace | `rocprofv3` | Which kernels dominate? | ~5–20% overhead |
| Deep-system | `rocprofiler-systems` | What is blocking the GPU? | High — use selectively |

The ladder is sequential. Start with light. Move to deep-trace only if the GPU looks busy but throughput is unexpectedly low. Move to deep-system only if kernel profiles look reasonable but something still stalls.

## LUMI-G and Toolchain Basics

A full LUMI-G node exposes 8 GPU-visible devices:

- 4 AMD MI250X modules
- 2 GCDs per MI250X
- 8 software-visible GCDs per node
- 56 CPU cores available to jobs

The ROCm profiling stack on LUMI:

| Tool | What it samples | Output |
|---|---|---|
| `rocm-smi` | GPU utilization, memory, power, temperature, clocks | `summary.json`, HTML report |
| `rocprofv3` | HIP kernel dispatches and durations | kernel summary CSV, trace artifacts |
| `rocprofiler-systems` | CPU threads, GPU kernels, memory transfers, OS events | Perfetto `.proto` trace |

This tutorial covers single-node profiling. For multi-node variants, see the `--nodes=2` scripts in [lumi-job-profiler/examples](https://github.com/aniskhan25/lumi-job-profiler/tree/main/examples).

## Dependencies

The profiling hook and analysis tools live in [lumi-job-profiler](https://github.com/aniskhan25/lumi-job-profiler). Clone it to scratch before submitting any job:

```bash
git clone https://github.com/aniskhan25/lumi-job-profiler.git /scratch/project_462000131/$USER/lumi-job-profiler
```

The profiling hook and analysis tools live there. The demo workloads are included in this repository under `examples/`.

## Setup

Run from the repository root on LUMI:

```bash
cd /path/to/profile-aware-ai
```

The Slurm jobs are configured for:

```text
project_462000131
/appl/local/laifs/containers/lumi-multitorch-latest.sif
```

Override either value by setting `SBATCH_ACCOUNT` or `LUMI_CONTAINER_IMAGE` in your environment before submitting.

Slurm logs are written to `logs/`. Profile artifacts are written to a per-job directory under `$SCRATCH`, reported at the end of each Slurm log.

## Part I: Light Profile — Is My GPU Busy?

> Is the GPU being kept busy, and at what memory and power level?

Light mode runs a `rocm-smi` process in parallel with your job, sampling GPU metrics at a fixed interval. It adds negligible overhead and works in every deployment scenario: containers, bare metal, single GPU, multi-GPU.

Submit the single-GCD profiling job:

```bash
sbatch jobs/run_light_1gcd.sh
```

At the end of the Slurm log, the job prints the paths to the generated profile artifacts.

Expected output shape:

```text
done iters=... elapsed=60.0s
Profile summary: /scratch/project_462000131/<user>/lumi-profile/<jobid>/summary.json
Profile analysis: /scratch/project_462000131/<user>/lumi-profile/<jobid>/analysis.json
Profile report: /scratch/project_462000131/<user>/lumi-profile/<jobid>/report.md
Profile report: /scratch/project_462000131/<user>/lumi-profile/<jobid>/report.html
```

Read `report.md` directly on LUMI to review the findings.

Interpretation:

| Signal | Value | What it means | Next action |
|---|---|---|---|
| `util_mean` | > 80% | GPU is well-utilized | Check throughput — if still low, go to Part III |
| `util_mean` | 50–80% | Moderate utilization | Look at data pipeline and batch size |
| `util_mean` | < 50% | GPU often idle | Fix data pipeline before profiling deeper |
| `mem_used_mean_gb` | Near device limit | Memory pressure | Reduce batch size or check for leaks |
| `power_mean_w` | Near TDP | GPU working hard | Expected for compute-bound workloads |
| `power_mean_w` | Far below TDP | GPU underloaded | Investigate idle periods |

> [!NOTE]
> Light mode tells you *what* the GPU was doing — not *why*. If utilization is high and throughput is still lower than expected, move to Part III to identify which kernels are consuming time.

> [!TIP]
> If `util_mean` is below 50%, diagnose the data pipeline or CPU-side preparation before going deeper. Adding more GCDs to an underutilized single GCD will not help.

## Part II: Light Profile Across Multiple GCDs

> Are all GCDs being used equally, or is work concentrated on one rank?

Submit the 8-GCD light profiling job:

```bash
sbatch jobs/run_light_8gcd.sh
```

Expected output shape:

```text
Profile summary: /scratch/project_462000131/<user>/lumi-profile/<jobid>/summary.json
Profile analysis: /scratch/project_462000131/<user>/lumi-profile/<jobid>/analysis.json
Profile report: /scratch/project_462000131/<user>/lumi-profile/<jobid>/report.md
Profile report: /scratch/project_462000131/<user>/lumi-profile/<jobid>/report.html
```

A healthy multi-GPU job shows similar `util_mean` values across all GCDs. Spread in those values is a signal worth acting on before scaling further.

Interpretation:

| Signal | Pattern | Likely cause | Next action |
|---|---|---|---|
| `util_mean` | Similar across all GPUs | Balanced load | Check aggregate throughput |
| `util_mean` | One GPU far higher | Rank 0 aggregation or uneven sharding | Review scatter/gather and data distribution |
| `util_mean` | All GPUs low | Data starvation or serialized launches | Fix data pipeline before scaling |
| `mem_used_mean_gb` | One GPU much higher | Uneven model placement | Check model parallelism or embedding placement |

> [!WARNING]
> Do not scale to more nodes if GPU utilization is already uneven across GCDs on a single node. Inter-node communication will widen the imbalance, not fix it.

## Part III: Deep-Trace — Which Kernels Dominate?

> The GPU looks busy, but throughput is still lower than expected. Where is the time actually going?

Deep-trace mode runs `rocprofv3` to capture HIP kernel dispatches and their durations. Use it when light mode shows healthy utilization but your throughput metric is below expectation.

> [!NOTE]
> Deep-trace requires `rocprofv3` inside the container. The job scripts in this tutorial handle that automatically using `lumi-multitorch-latest.sif`. For your own container, confirm `rocprofv3` is present before switching to deep-trace mode.

Submit the deep-trace job:

```bash
sbatch jobs/run_deep_trace_8gcd.sh
```

Expected output shape at the end of the Slurm log:

```text
Deep profile summary: /scratch/project_462000131/<user>/lumi-profile/<jobid>/deep_profile/trace/summary.json
Deep trace manifest:  /scratch/project_462000131/<user>/lumi-profile/<jobid>/deep_profile/deep_manifest.json
Profile summary:      /scratch/project_462000131/<user>/lumi-profile/<jobid>/summary.json
Profile analysis:     /scratch/project_462000131/<user>/lumi-profile/<jobid>/analysis.json
Profile report:       /scratch/project_462000131/<user>/lumi-profile/<jobid>/report.md
Profile report:       /scratch/project_462000131/<user>/lumi-profile/<jobid>/report.html
```

The deep trace summary contains per-rank kernel stats, top HIP API calls, and top kernel dispatches aggregated across all ranks. The raw per-rank trace CSVs are under `deep_profile/trace/raw/<node>/rank-N/`.

Interpretation:

| Signal | Pattern | What it means | Next action |
|---|---|---|---|
| Top kernel | One GEMM > 60% of kernel time | Compute-bound — expected | Verify batch size and arithmetic intensity |
| Top kernel | Many small kernels, each < 1% | Launch fragmentation | Fuse operations or increase per-kernel work |
| Top kernel | Memory-copy kernels high | Data movement dominant | Review host–device transfer patterns |
| Mean dispatch | < 10 µs per kernel | Kernels too small | Increase tile sizes or use larger ops |
| Total kernel time | Much less than walltime | Large gap outside kernels | Go to Part IV to find what fills the gap |

> [!TIP]
> If total kernel time accounts for most of walltime and the dominant kernel is a large GEMM, the workload is in a healthy state. No further profiling is needed unless you have a specific optimization target.

> [!WARNING]
> Deep-trace overhead is typically 5–20%. Do not submit large production jobs with this mode enabled. Use a short representative run of 30–120 seconds.

## Part IV: Deep-System — What Is Blocking the GPU?

> Kernel profiles look reasonable, but something is still stalling execution. What is happening at the system level?

Deep-system mode runs `rocprofiler-systems` to produce a Perfetto-compatible trace covering CPU threads, GPU kernels, memory copies, and OS-level events simultaneously. Use it as a last resort when the previous two levels have not identified the bottleneck.

Submit the deep-system job:

```bash
sbatch jobs/run_deep_system_8gcd.sh
```

Expected output shape at the end of the Slurm log:

```text
Deep system summary:  /scratch/project_462000131/<user>/lumi-profile/<jobid>/deep_profile/system/summary.json
Profile summary:      /scratch/project_462000131/<user>/lumi-profile/<jobid>/summary.json
Profile analysis:     /scratch/project_462000131/<user>/lumi-profile/<jobid>/analysis.json
Profile report:       /scratch/project_462000131/<user>/lumi-profile/<jobid>/report.md
Profile report:       /scratch/project_462000131/<user>/lumi-profile/<jobid>/report.html
```

The raw per-rank Perfetto traces are under `deep_profile/system/raw/<node>/rank-N/`. Load any `.proto` file in [Perfetto](https://ui.perfetto.dev).

> [!NOTE]
> deep-system uses the ROCm 6.4 container (`lumi-multitorch-full-u24r64f21m43t29-*`) because rocprofiler-systems was built against ROCm 6.x sonames. The other profiling modes use `lumi-multitorch-latest.sif`.

What to look for:

| Signal | What it means | Next action |
|---|---|---|
| Long CPU gap before GPU kernel | Host-side preparation or Python overhead | Profile Python, reduce pre-kernel work |
| GPU idle between kernels | Sequential launch bottleneck | Use HIP streams or async dispatch |
| Large `hipMemcpy` blocks | Frequent host–device data transfers | Pin memory, prefetch, or keep data on GPU |
| CPU thread contention | GIL hold or dataloader worker stall | Increase dataloader workers or use persistent workers |

> [!WARNING]
> Deep-system profiling has high overhead and produces large trace files (~1 GB per rank per 30 seconds). Use at most 10–15 seconds. Only reach for this mode when light and deep-trace have not identified the bottleneck.

## Reading the Reports

Every profiling run produces three output files regardless of mode.

**`summary.json`**

Compact JSON with per-GPU aggregates: mean and max utilization, memory used, power, temperature, and clock frequencies. Use this for quick programmatic comparisons across runs.

**`analysis.json`**

Rule-based findings derived from `summary.json`. Each finding has a severity, a signal name, and a plain-text description:

```json
{
  "findings": [
    {
      "severity": "warning",
      "signal": "low_gpu_utilization",
      "detail": "Mean GPU utilization below 50% on GPU 0. Job may be data-starved or launch-bound."
    }
  ]
}
```

**`report.md` / `report.html`**

The markdown report is readable directly on LUMI with `cat` or any pager. The HTML version contains the same content with plots — copy it to your local machine to open in a browser.

## Repository Layout

```text
examples/   Demo workloads (single-GPU and distributed)
jobs/       Slurm job scripts for each profiling level
logs/       Slurm output directory (created on first submission)
```

Profiling infrastructure lives in [lumi-job-profiler](https://github.com/aniskhan25/lumi-job-profiler). Clone it to `$SCRATCH` before submitting jobs.
