# External rollout vLLM prototype

This prototype runs async Single Controller GRPO with rollout generation in a
vLLM server outside NeMo-RL's training Ray cluster.

The launch topology is:

```text
one Slurm heterogeneous job
├── hetgroup 0: 4-node Megatron policy + 1-node Gym safety judge
└── hetgroup 1: policy rollout vLLM + GenRM + NL2Bash pools
```

A site-specific launcher can reuse
`tools/external_gym_vllm/run_in_allocation.sh`. It should start the external
pools, wait for health, substitute their OpenAI base URLs into the training
command, launch `ray.sub` only on the training hetgroup, and tear down both
sides when either exits.

## Stock vLLM API usage

No endpoint plugin is required. The launcher sets `VLLM_SERVER_DEV_MODE=1`,
which enables vLLM's development control endpoints. The controller uses:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Verify the engine is live. |
| `GET` | `/v1/models` | Verify the served model alias. |
| `GET` | `/server_info` | Verify development endpoints are enabled. |
| `POST` | `/pause?mode=keep&clear_cache=true` | Freeze in-flight work and clear stale KV state before refit. |
| `POST` | `/collective_rpc` | Invoke `reload_weights(weights_path=...)` on all engine workers. |
| `POST` | `/resume` | Resume request processing after a successful reload. |

NeMo-Gym sends rollout requests directly to the server's standard
OpenAI-compatible API. During refit, Megatron-Bridge exports the live policy to
a unique HF checkpoint directory on shared storage. vLLM then reloads that
directory globally through `/collective_rpc`.

The `keep` pause mode preserves pending HTTP requests across the refit. Cache
clearing moves active requests back to vLLM's waiting queue, so they recompute
their prefix with the new weights after `/resume` rather than retaining stale
KV state.

These development endpoints are powerful and must remain on the job's private
network. A failed reload leaves the server paused because some workers may
already contain the new version.

## Launching the Nano 3.5 RLVR smoke

The smoke configuration is
`examples/nemo_gym/nemotron-3.5-nano/rlvr_sc_smoke_small_external_vllm.yaml`.
A launcher must supply deployment-specific artifact locations rather than
placing them in Python or YAML:

```bash
export EXTERNAL_ROLLOUT_VLLM_URL=http://rollout-service:8000/v1
export EXTERNAL_ROLLOUT_HF_EXPORT_DIR=/shared/path/to/hf_exports

uv run examples/run_grpo_external_vllm_single_controller.py \
  --config examples/nemo_gym/nemotron-3.5-nano/rlvr_sc_smoke_small_external_vllm.yaml \
  policy.model_name=/shared/path/to/policy \
  data.train.data_path=/shared/path/to/train.jsonl \
  data.validation.data_path=/shared/path/to/validation.jsonl
```

Container images, optional prebuilt Gym environments, reward-model checkpoints,
and cluster scheduler settings likewise belong in the site-specific launcher.
External vLLM servers can use the prebuilt `VllmAsyncGenerationWorker` virtual
environment from the NeMo RL image.

HF refit exports default to `$BASE_LOG_DIR/hf_exports`. Override
`EXTERNAL_ROLLOUT_HF_EXPORT_DIR` when needed. That path must be mounted at the
same absolute location in both containers. Version directories are retained
for this prototype so vLLM never reloads an overwritten path.

## Scaling boundary

One native vLLM deployment may contain multiple TP/PP/DP workers;
`/collective_rpc` applies to every worker owned by that engine. A separate
router in front of multiple independent `vllm serve` deployments does not make
control calls global. That later topology needs a control-plane fan-out to
each backend, even if Gym continues to see one inference URL.

Preflight recognizes the backend count reported by the repository's existing
external-vLLM load balancer and rejects counts other than one. Native vLLM
parallelism inside that one backend is unaffected.
