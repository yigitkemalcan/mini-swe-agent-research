# Agentic AI Characterization — System and Environment Overview

> **Purpose:** This file gives an agent the stable machine, model, environment, and folder context needed to work on the project.
>
> **Current reset state:** `/mnt/raid0/jovans/yigit/agent-characterization/mini-swe-agent` is a fresh checkout of the original mini-SWE-Agent code at commit `04d809ceab9df28f9adaed044884180159172930`.

---

## 1. Research Goal

The project studies agentic AI workloads from a systems perspective, especially the alternation between:

- LLM serving on the GPU side, and
- agent/tool execution on the CPU / sandbox side.

The benchmark target is **SWE-Bench Verified**.

Current core components:

| Component                              | Current value                              |
| -------------------------------------- | ------------------------------------------ |
| Agent                                  | mini-SWE-Agent                             |
| Clean mini-SWE-Agent baseline commit   | `04d809ceab9df28f9adaed044884180159172930` |
| Model                                  | `Qwen/Qwen3-235B-A22B-Instruct-2507`       |
| Inference server                       | vLLM `0.10.1.1`                            |
| Benchmark                              | SWE-Bench Verified                         |
| Preferred characterization concurrency | `--workers 1`                              |

---

## 2. Remote Machine

| Item                 | Value                     |
| -------------------- | ------------------------- |
| SSH target           | `jovan@10.0.0.4`          |
| Hostname             | `vmss-a100000002`         |
| GPUs                 | 8 x NVIDIA A100-SXM4-80GB |
| NVIDIA driver        | `560.35.05`               |
| Driver-reported CUDA | `12.6`                    |
| CPU                  | AMD EPYC 7V12, 96 CPUs    |
| RAM                  | about 1.7 TiB             |
| Docker root          | `/mnt/raid0/docker`       |

The main project data is under `/mnt/raid0/jovans/yigit/agent-characterization`.

Large model/runtime assets are stored separately under `/mnt/jovans_models/experiment-yigit/qwen-experiment`.

---

## 3. Current Folder Structure

### 3.1 Main project area

```text
/mnt/raid0/jovans/yigit/agent-characterization/
├── mini-swe-agent/          # fresh upstream mini-SWE-Agent checkout
├── envs/
│   └── swebench-eval/       # environment used for official SWE-Bench evaluation
```

Important paths:

| Purpose                 | Path                                                                |
| ----------------------- | ------------------------------------------------------------------- |
| Project root            | `/mnt/raid0/jovans/yigit/agent-characterization`                    |
| mini-SWE-Agent checkout | `/mnt/raid0/jovans/yigit/agent-characterization/mini-swe-agent`     |
| SWE-Bench evaluator env | `/mnt/raid0/jovans/yigit/agent-characterization/envs/swebench-eval` |

There is also a compatibility symlink:

```text
/home/jovan/agent-characterization
    -> /mnt/raid0/jovans/yigit/agent-characterization
```

Some older environments or entry points may still reference this path, so do not remove it casually.

### 3.2 Model and vLLM area

The model-serving environment and caches are **outside** the agent-characterization checkout:

```text
/mnt/jovans_models/experiment-yigit/qwen-experiment/
├── envs/
│   └── vllm/
├── huggingface-cache/
├── vllm-cache/
└── uv-cache/
```

Important paths:

| Purpose            | Path                                                                    |
| ------------------ | ----------------------------------------------------------------------- |
| vLLM environment   | `/mnt/jovans_models/experiment-yigit/qwen-experiment/envs/vllm`         |
| Hugging Face cache | `/mnt/jovans_models/experiment-yigit/qwen-experiment/huggingface-cache` |
| vLLM cache         | `/mnt/jovans_models/experiment-yigit/qwen-experiment/vllm-cache`        |
| uv cache           | `/mnt/jovans_models/experiment-yigit/qwen-experiment/uv-cache`          |

The pinned Qwen snapshot previously used is:

```text
/mnt/jovans_models/experiment-yigit/qwen-experiment/huggingface-cache/hub/models--Qwen--Qwen3-235B-A22B-Instruct-2507/snapshots/ac9c66cc9b46af7306746a9250f23d47083d689e
```

The snapshot is large (about 438 GiB), so it should not be duplicated into the mini-SWE-Agent repository.

---

## 4. vLLM / Model Software State

Recorded serving stack:

| Component        | Version       |
| ---------------- | ------------- |
| vLLM             | `0.10.1.1`    |
| Transformers     | `4.55.2`      |
| Tokenizers       | `0.21.4`      |
| PyTorch          | `2.7.1+cu126` |
| Hugging Face Hub | `0.36.2`      |

Model:

```text
Qwen/Qwen3-235B-A22B-Instruct-2507
```

The intended serving setup uses all 8 A100 GPUs with vLLM.

At this reset point, do **not** assume that any previous custom vLLM instrumentation is installed or active. Verify the actual vLLM environment before relying on custom logging behavior.

---

## 5. Runtime Architecture

The basic architecture is:

```text
Remote host
│
├── mini-SWE-Agent controller
│   └── /mnt/raid0/jovans/yigit/agent-characterization/mini-swe-agent
│
├── SWE-Bench execution environment
│   └── Docker-based task/container environment used by mini-SWE-Agent
│
└── vLLM server
    ├── environment:
    │   /mnt/jovans_models/experiment-yigit/qwen-experiment/envs/vllm
    ├── model:
    │   Qwen/Qwen3-235B-A22B-Instruct-2507
    └── GPUs:
        8 x A100-SXM4-80GB
```

Conceptually:

```text
mini-SWE-Agent
   ├── sends model requests -> vLLM -> Qwen inference on GPUs
   └── sends tool commands  -> SWE-Bench Docker environment
```

The agent controller and the SWE-Bench tool-execution environment are separate components.

---

## 6. Evaluation Environment

The official SWE-Bench evaluation environment is still kept at:

```text
/mnt/raid0/jovans/yigit/agent-characterization/envs/swebench-eval
```

This environment is separate from:

- the clean mini-SWE-Agent checkout, and
- the vLLM environment.

Evaluation should remain conceptually separate from agent execution and model serving.

---

## 7. Storage Notes

`/mnt/raid0` is used for the project checkout, run artifacts, Docker storage, and evaluation artifacts.

`/mnt/jovans_models/experiment-yigit/qwen-experiment` contains the model-serving environment and large model/cache data.

The model snapshot and other large assets should not be copied into the Git repository.
