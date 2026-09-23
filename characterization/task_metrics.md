# Per-task resources and per-step timings

Run the normal setup_scripts/run-swebench.sh command with CPU sampling enabled
(default: 0.1 seconds). Docker stats is optional: --no-docker-stats disables it
without affecting these measurements. The runner generates task_metrics/tasks.csv,
steps.csv and analysis.json after stopping collection. To repeat analysis:

    python characterization/analyze_task_metrics.py RUN_DIRECTORY

Generate empirical CDFs after analysis:

    python characterization/visual/plot_task_metrics.py RUN_DIRECTORY

This writes task_metrics/plots/task_steps_cdf, task_resources_cdf and step_times_cdf
in PNG format, plus cdf_summary.json with exact CDF coordinates/counts.
Task resource plots include separate average/peak curves for CPU and memory.
Every task has equal weight in task plots; every timed step has equal weight in
timing plots, including failed model calls and zero-tool steps. Nonfinite values
are excluded per series and counted in the JSON. A one-task run produces single
jumps in task-level CDFs; multiple tasks are needed for a meaningful distribution.

The host CPU count is recorded in system_metrics/meta.json. For older metadata,
pass --host-cpus N explicitly. Analysis requires container_samples.csv; this
implementation assumes resource collection runs alongside the instances.

## Per task

- steps: existing model_stats.api_calls (attempted model calls).
- cpu_avg_percent and cpu_peak_percent: container CPU utilization as a percentage
  of ALL host CPUs. One fully busy CPU on a 96-CPU host is about 1.04%.
- memory_avg_mib and memory_peak_mib: Docker-compatible cache-adjusted footprint,
  not percentage or process RSS. Uses total_inactive_file on this cgroup v1 host.
- resource_coverage_s: duration actually covered by the sampled counter intervals.
- task_duration_s: environment startup through agent completion/failure.
- environment_setup_s: environment creation plus configured startup commands.
- inference_total_s and tool_total_s: sums of the per-step call durations.

The full container ID in each trajectory matches its cgroup samples exactly.
This works with concurrent instances; no image/name/order inference is used.
The task window begins before environment setup and ends when the agent returns
or fails, excluding delayed container teardown. Setup before a container exists
has no container samples and is not treated as zero resource use.

CPU averages are time-weighted cumulative-counter delta rates. Memory averages
hold each gauge to the next sample. Only interval portions within the task window
are weighted. Boundary CPU intervals assume constant utilization. No duration is
invented beyond the last sample. Peaks are sampled maxima; short bursts can be
missed. Resource status and coverage expose absent/insufficient samples.

## Per step

Inference time wraps the entire model.query(messages) call, including request,
response, parsing and any internal retries. Tool time sums wrappers around each
env.execute(action), including command preparation, communication, execution and
return processing inside the environment call. These are elapsed wall-clock times,
not pure GPU-kernel or pure shell-process time.

action_count counts attempted execute calls, not shell statements. A failed or
final-submission call is included; subsequent unexecuted actions are not. A step
whose model call fails has zero tool actions. A limit check preventing the next
model call does not create a new step. Exception outcomes are retained, while the
original exception propagates normally. Returned tool error details stay in the
existing observation messages.

One-time environment creation is outside the step timers and reported separately.
Observation formatting, saving trajectories and other loop work are also outside
the wrappers, so setup + inference + tool time need not equal task duration.

Raw step_timings are saved in each trajectory, with monotonic timestamps sharing
the collector's host clock. Existing trajectories cannot supply missing historical
timings or exact task windows; those values are left unavailable. Environment
startup failures before an agent exists have no trajectory in the existing runner.
The analysis JSON also preserves system logger errors for coverage assessment.
