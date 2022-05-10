# \[RFC\] DietCode: An Auto-Scheduler for Dynamic Tensor Programs

## Motivation

Achieving high performance for compute-intensive operators in machine learning
workloads is a crucial but challenging task. Many machine learning and system
practitioners rely on vendor libraries or auto-schedulers to do the job. While
the former requires large engineering efforts, the latter only supports
static-shape workloads in existing works. It is difficult, if not impractical,
to apply existing auto-schedulers directly to dynamic-shape workloads, as this
leads to extremely long auto-scheduling time.

## Proposed Design

We observe that the key challenge faced by existing auto-schedulers when
handling a dynamic-shape workload is that they cannot construct a unified search
space for all the possible shapes of the workload, because their search space is
shape-dependent. To address this, we propose DietCode, a new auto-scheduler
framework that efficiently supports dynamic-shape workloads by constructing a
shape-generic search space and cost model. Under this construction, all shapes
jointly search within the same space and update the same cost model when
auto-scheduling, which is therefore more efficient compared with existing
auto-schedulers. We evaluate DietCode using state-of-the-art machine learning
workloads on a modern GPU.

### Framework Overview

<img src="./docs/figures/DietCode.jpg" width="61.8%" />

### User Interface

```Python
T, T_vals = tir.ShapeVar('T’), list(range(1, 128))

task = SearchTask(func=Dense, args=(16*T, 768, 2304),
                  shape_vars=(T,), wkl_insts=(T_vals,)
                  wkl_inst_weights=([1. for _ in T_vals],))

tune_option = TuningOptions(
                  num_measure_trials=auto_sched_ntrials,
                  runner=local_rpc_measure_ctx.runner,
                  measure_callbacks=[RecordToFile(sched_log_fname)]
              )
search_policy = SketchPolicy(search_task, XGBModel())

search_task.tune(tune_option, search_policy)
```

Note that the above interface is based on Ansor, and we will migrate the same
changes to the MetaScheduler accordingly.

## Evaluation Results

Our evaluation shows that DietCode has the following key strengths when
auto-scheduling an entire model end-to-end: 

1. reduces the auto-scheduling time by 5.88x less than the
state-of-the-art auto-scheduler on 8 uniformly sampled dynamic shapes, and

1. improves performance by up to 69.5% better than the auto-scheduler and 18.6%
better than the vendor library. All these advantages make DietCode an efficient
and practical solution for dynamic-shape workloads.

## Upstreaming Milestones

We propose the following milestones for upstreaming, where each bullet point
corresponds to a PR of roughly several hundred lines.

- [ ] Code Generation Support
  - Local Padding
  - Loop Partitioning
- [ ] Auto-Scheduler
  - Frontend Interface
  - Sketch Generation
  - Random Annotations
  - Program Measurer
  - Micro-Kernel Cost Model
  - Evolutionary Search
- [ ] Dynamic-Shape Program Compilation
  ```CUDA
  __global__ void default_function(float* X, float* Y, float* Z,
                                   const int T) {
                                   // Note the extra `T` here
  }
  ```
- [ ] Decision-Tree Dispatching
