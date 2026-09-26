+# Effective configuration comparison: job 7389614 vs W&B bbix10jf

Inputs:

- `used_nemo_rl_config.yaml`: exact NeMo-RL recipe copied into job 7389614.
- `resolved_megatron_config.yaml`: rank-zero resolved Megatron-Bridge container emitted by that job.
- W&B target: `adlr/megatron-vlm-v3/bbix10jf`, queried through the W&B API.

## Merge rule

For a same-named setting that is semantically part of the Megatron/Bridge container, the saved Megatron value is the effective value. NeMo-RL-only settings remain NeMo-RL settings. The raw sections below use same-name matching as an audit aid; a matching leaf name does not always mean identical runtime semantics.

## Raw NeMo-RL vs saved-Megatron disagreements (3)

```text
CONFLICT	empty_unused_memory_level	2	nemo_rl.policy.megatron_cfg.empty_unused_memory_level	0	saved_megatron.train.empty_unused_memory_level
CONFLICT	radio_force_eval_mode	false	nemo_rl.policy.megatron_cfg.radio_force_eval_mode	true	saved_megatron.model.radio_force_eval_mode
CONFLICT	recompute_vision	true	nemo_rl.policy.megatron_cfg.recompute_vision	false	saved_megatron.model.recompute_vision
```

## Effective same-named values that differ from bbix10jf (70)

```text
DIFF	attention_backend	5	saved_megatron.model.attention_backend	"flash"
DIFF	attention_softmax_in_fp32	true	saved_megatron.model.attention_softmax_in_fp32	false
DIFF	calculate_per_token_loss	true	saved_megatron.model.calculate_per_token_loss	false
DIFF	chat_template	"default"	nemo_rl.policy.tokenizer.chat_template	null
DIFF	context_parallel_size	8	saved_megatron.model.context_parallel_size	16
DIFF	cross_entropy_loss_fusion	true	saved_megatron.model.cross_entropy_loss_fusion	false
DIFF	data_parallel_sharding_strategy	"optim_grads_params"	saved_megatron.ddp.data_parallel_sharding_strategy	"no_shard"
DIFF	dist_ckpt_strictness	"assume_ok_unexpected"	saved_megatron.checkpoint.dist_ckpt_strictness	"log_unexpected"
DIFF	distributed_timeout_minutes	10	saved_megatron.dist.distributed_timeout_minutes	120
DIFF	embedding_init_method_std	0.02	saved_megatron.model.embedding_init_method_std	null
DIFF	eval_iters	100	saved_megatron.validation.eval_iters	0
DIFF	exit_signal	15	saved_megatron.train.exit_signal	"SIGTERM"
DIFF	finetune	true	saved_megatron.checkpoint.finetune	false
DIFF	first_last_layers_bf16	true	saved_megatron.model.first_last_layers_bf16	false
DIFF	freeze_sound_projection	true	saved_megatron.model.freeze_sound_projection	false
DIFF	gradient_accumulation_fusion	false	saved_megatron.model.gradient_accumulation_fusion	true
DIFF	high_priority_stream_groups	[]	saved_megatron.dist.high_priority_stream_groups	["ep"]
DIFF	inference_cuda_graph_scope	1	saved_megatron.model.inference_cuda_graph_scope	"none"
DIFF	init_method_std	0.02	saved_megatron.model.init_method_std	0.014
DIFF	language_model_type	"nemotron6-moe"	saved_megatron.model.language_model_type	"nemotron6-super"
DIFF	local_rank	0	saved_megatron.dist.local_rank	3
DIFF	log_interval	100	saved_megatron.logger.log_interval	10
DIFF	log_num_zeros_in_grad	false	saved_megatron.optimizer.log_num_zeros_in_grad	true
DIFF	log_params_norm	false	saved_megatron.logger.log_params_norm	true
DIFF	log_progress	false	saved_megatron.logger.log_progress	true
DIFF	log_throughput	false	saved_megatron.logger.log_throughput	true
DIFF	log_timers_to_tensorboard	false	saved_megatron.logger.log_timers_to_tensorboard	true
DIFF	logging_level	0	saved_megatron.logger.logging_level	20
DIFF	lr_decay_iters	1000	saved_megatron.scheduler.lr_decay_iters	null
DIFF	lr_warmup_init	4.9999e-06	saved_megatron.scheduler.lr_warmup_init	0
DIFF	lr_warmup_iters	3	saved_megatron.scheduler.lr_warmup_iters	0
DIFF	lr_warmup_samples	0	saved_megatron.scheduler.lr_warmup_samples	9100
DIFF	manual_gc	false	saved_megatron.train.manual_gc	true
DIFF	manual_gc_interval	0	saved_megatron.train.manual_gc_interval	10
DIFF	memory_snapshot_path	"/nemo_run/snapshot.pickle"	saved_megatron.profiling.memory_snapshot_path	"snapshot.pickle"
DIFF	microbatch_group_size_per_vp_stage	1	saved_megatron.model.microbatch_group_size_per_vp_stage	null
DIFF	moe_router_group_topk	1	saved_megatron.model.moe_router_group_topk	null
DIFF	moe_router_num_groups	1	saved_megatron.model.moe_router_num_groups	null
DIFF	mtp_hybrid_override_pattern	"*E"	saved_megatron.model.mtp_hybrid_override_pattern	null
DIFF	muon_momentum	0.95	saved_megatron.optimizer.muon_momentum	0.9
DIFF	muon_tp_mode	"duplicated"	saved_megatron.optimizer.muon_tp_mode	"blockwise"
DIFF	num_workers	1	nemo_rl.data.energon.num_workers	4
DIFF	optimizer_offload_fraction	0.0	saved_megatron.optimizer.optimizer_offload_fraction	1
DIFF	override_opt_param_scheduler	false	saved_megatron.scheduler.override_opt_param_scheduler	true
DIFF	pad_vocab_size	false	saved_megatron.tokenizer.pad_vocab_size	true
DIFF	pretrained_checkpoint	"/lustre/fsw/portfolios/coreai/users/guyueh/hf_home/nemo_rl/model__lustre_fsw_portfolios_coreai_users_rohitkumarj_data_models_NVIDIA-Nemotron-3.5-Super-midtrain-67B-vision-pretrained"	saved_megatron.checkpoint.pretrained_checkpoint	"/lustre/fs1/portfolios/nemotron/projects/nemotron_omni_vision/users/matthieul/workspace/output/super35_dualds_4k_1of8_recover_iter366_pre_sft_radio_v4_h_vision_adapter_pretrain_1377_dss_video_embedder/checkpoints/torch_dist"
DIFF	profile_ranks	[0]	saved_megatron.profiling.profile_ranks	[]
DIFF	radio_force_eval_mode	true	saved_megatron.model.radio_force_eval_mode	false
DIFF	rank	0	saved_megatron.tokenizer.rank	1023
DIFF	recompute_vision	false	saved_megatron.model.recompute_vision	true
DIFF	rerun_mode	"disabled"	saved_megatron.rerun_state_machine.rerun_mode	"validate_results"
DIFF	seq_length	262144	saved_megatron.model.seq_length	12288
DIFF	shuffle_buffer_size	1000	nemo_rl.data.energon.shuffle_buffer_size	100
DIFF	sound_model_type	"parakeet"	saved_megatron.model.sound_model_type	null
DIFF	split	"train"	nemo_rl.data.train.split	"969, 30, 1"
DIFF	strict_fsdp_dtensor_load	false	saved_megatron.checkpoint.strict_fsdp_dtensor_load	true
DIFF	timing_log_level	0	saved_megatron.logger.timing_log_level	1
DIFF	tokenizer_model	"/lustre/fsw/portfolios/coreai/users/rohitkumarj/data/models/NVIDIA-Nemotron-3.5-Super-midtrain-67B-vision-pretrained"	saved_megatron.tokenizer.tokenizer_model	"/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/tpoon/repos_nanov35/resources/tokenizer/nano_v35_sft_v10_closethink_unmask_orig6k_vlm"
DIFF	tokenizer_type	"nemotron6-moe"	saved_megatron.model.tokenizer_type	"MultimodalTokenizer"
DIFF	train_iters	1000000	saved_megatron.train.train_iters	null
DIFF	trust_remote_code	true	saved_megatron.tokenizer.trust_remote_code	false
DIFF	use_checkpoint_args	false	saved_megatron.checkpoint.use_checkpoint_args	true
DIFF	use_cpu_initialization	false	saved_megatron.model.use_cpu_initialization	null
DIFF	use_fused_weighted_squared_relu	false	saved_megatron.model.use_fused_weighted_squared_relu	true
DIFF	use_gloo_process_groups	false	saved_megatron.dist.use_gloo_process_groups	true
DIFF	use_persistent_ckpt_worker	true	saved_megatron.checkpoint.use_persistent_ckpt_worker	false
DIFF	use_precision_aware_optimizer	true	saved_megatron.optimizer.use_precision_aware_optimizer	false
DIFF	use_tokenizer_model_from_checkpoint_args	true	saved_megatron.checkpoint.use_tokenizer_model_from_checkpoint_args	false
DIFF	video_decode_thread_count	4	nemo_rl.data.energon.task_encoder.options.video_decode_thread_count	0
DIFF	vocab_size	131072	saved_megatron.model.vocab_size	null
```

## Interpretation notes

- `train.empty_unused_memory_level` in the saved Bridge container is not the same runtime setting as `policy.megatron_cfg.empty_unused_memory_level`; NeMo-RL workers read the latter directly.
- The SFT setup path forwards `radio_force_cpe_eval_mode` and tower-freeze controls, but does not forward the recipe's `radio_force_eval_mode` or `recompute_vision`/`vision_recompute_*` keys. Their saved values are therefore provider defaults, not the recipe's requested values.
- `local_rank`, `rank`, checkpoint paths, profiler paths, and log paths are run-specific rather than model-hyperparameter differences.
- The serializer currently writes enum values numerically. `attention_backend: 5` therefore cannot be compared textually with bbix10jf's `flash` without resolving the enum from the matching Megatron-Core version.
