# Megatron Energon Dataloader Parity

This note records the NeMo-RL changes used to make the packed Nemotron
multimodal SFT Energon dataloader match the Megatron-LM reference dataloader at
the model-forward boundary.

The comparison target was stricter than ordinary training equivalence. With the
same source YAML, checkpoint/tokenizer, packing buffer size, dataloader seed, and
one logical data-parallel rank, the probe compared:

- token IDs
- token loss masks
- packed sequence metadata
- packed sample lengths and source IDs
- attached visual/audio tensors where applicable

Some changes below fix real semantic differences. Others are deterministic
ordering controls that make a probe reproducible but do not change how an
individual sample is encoded.

## Reference Contract

Megatron's multimodal Energon path uses compact media sentinels in `input_ids`.
For images, the sentinel is `-200`, not the tokenizer's positive `"<image>"`
vocabulary ID. The model expands that compact sentinel into the projected image
embedding span inside the forward preprocessing path.

That means a dataloader sample containing one image should reach the model as a
compact stream such as:

```text
[text_a, -200, text_b]
```

It should not reach the model as:

```text
[text_a, <image>, <image>, ..., <image>, text_b]
```

The latter is an already-expanded token stream. It may be a valid contract for a
different model path, but it is not the Megatron-LM reference contract used for
this parity target.

## Functional Mismatch Fixes

### Compact Image Sentinel

NeMo-RL previously spliced the tokenizer's positive `"<image>"` ID and expanded
visual placeholders in post-encoding. The patch changes Nemotron tokenization to
splice `-200` and removes dataloader-side visual placeholder expansion.

This is a functional fix because it changes the sample presented to the model:

- `input_ids` now carry Megatron's compact negative image sentinel.
- Literal `"<image>"` text no longer collides with media placeholders.
- Sequence length at the dataloader boundary now matches Megatron.
- Packing cost, labels, loss masks, and position/attention metadata are derived
  under the same compact-sentinel contract.

The tokenizer still rejects the audio sentinel text `"<so_embedding>"`, because
the audio path scans a positive tokenizer ID and can still collide with literal
source text.

### Packed Collation Metadata

Packed Energon samples have both source lengths and padded source lengths. The
Megatron reference pads each source sample to its own padded length and builds
cumulative boundaries from those per-source padded lengths.

NeMo-RL previously appended tail padding to the whole pack and forced the final
`cu_seqlens_padded` boundary to the pack capacity. That can make the last sample
look as if it owns all remaining capacity, which changes packed attention and
position metadata.

The patch now:

- pads each source to its own padded length
- builds `cu_seqlens` and `cu_seqlens_padded` from per-source lengths
- leaves unused pack capacity as batch padding rather than last-sample padding
- validates that a physical pack does not exceed its capacity

For example, with capacity `128` and two sources `5 -> 8` and `7 -> 8`:

```text
before:
  tokens:             A(5), B(7), tail_pad(116)
  cu_seqlens_padded:  [0, 8, 128]

after:
  tokens:             A(5), pad(3), B(7), pad(1)
  cu_seqlens_padded:  [0, 8, 16]
```

This is a functional fix because the packed-attention metadata changes the
layout consumed by the model.

### Nemotron Pad Token

For the Nemotron tokenizer used in this path, Megatron uses the tokenizer
`<unk>` token as the pad ID. The patch adds a small helper that detects the
Nemotron message token layout and chooses `<unk>` as the packed-padding token,
falling back to `tokenizer.pad_token_id` for other tokenizers.

This is a functional fix when the tokenizer's default pad ID differs from the
Megatron pad ID, because padding tokens participate in input IDs and packed
metadata.

### Auxiliary Media Basename Fallback

Megatron's cooker resolves media through `aux_data_prefixes`. If an absolute
media path matches an auxiliary data prefix but is not locally readable, it
falls back to the basename and relies on the auxiliary media source.

The patch adds the same opt-in behavior to NeMo-RL's auxiliary media resolver
for call sites that accept missing absolute paths.

This is a functional fix for affected samples. Without it, NeMo-RL can fail to
load media or load a different path than Megatron, producing different visual
features.

## Probe Determinism And Stream Alignment

### Balanced Greedy Knapsack

Megatron's `balanced_greedy_knapsack`:

1. sorts samples by padded length descending
2. creates `ceil(total_length / capacity) + delta` empty bins
3. repeatedly tries the current least-loaded bin
4. appends a new empty bin if the next sample does not fit

NeMo-RL's previous implementation selected among all fitting candidate bins and
did not plumb `balanced_knapsack_delta` through the factory path. The patch
matches Megatron's bin-selection loop and passes `balanced_knapsack_delta` for
the balanced greedy packer.

This primarily affects which samples share a physical pack and the order of
packs. That is usually statistical drift, not a change to an individual sample's
tokenization. It is still required for bit-for-bit probe parity.

### Task Encoder Packing Config Plumbing

Some configs carry physical Energon packing settings under
`data.energon.task_encoder.packing`, including:

- `name`
- `buffer_size`
- `options.max_sequence_length`
- `options.sequence_length_pad_multiple`
- `options.balanced_knapsack_delta`

The loader now reads these settings and applies them to the physical Energon
packer when the top-level loader fields are absent. This makes the effective
dataloader identity explicit and prevents a config from silently ignoring the
packing settings used by the Megatron reference.

### Train Dataset Keyword Arguments

For the packed training dataloader, NeMo-RL now mirrors the Megatron
`get_train_dataset` call more closely. The parity patch removes NeMo-only
training kwargs that changed stream, epoch, or tail behavior:

- `split_part`
- `batch_drop_last=True`
- `shuffle_over_epochs_multiplier=1`
- `virtual_epoch_length`

These are mostly stream-alignment fixes. If the same samples eventually appear,
they are statistical/order drift. If a removed option changes dataset
membership, such as dropping tail samples or selecting a different split, it
becomes a real dataset-membership mismatch.

### Pack Shuffle Seed

Megatron shuffles selected packs using Python's ambient RNG under Energon's
stateless seed restoration. NeMo-RL previously also used ambient
`random.shuffle`, but framework-specific prior random calls can still shift the
pack order.

The patch adds optional `pack_shuffle_seed`. When unset, production behavior is
unchanged. When set by a probe, NeMo-RL uses a local `random.Random(seed)` to
make pack order deterministic.

This is a probe-control knob. It should not be required for semantic training
equivalence.

## Validation

The parity probe that motivated these changes compared Megatron-LM and NeMo-RL
with:

- one logical data-parallel rank
- the same data YAML
- the same packing buffer size
- the same dataloader seed
- the same checkpoint/tokenizer source
- matching sequence-length pad multiple

After the patches, the probe matched token IDs, token loss masks, packed
sequence metadata, and packed source sample lengths for the inspected batches.

## Change Classification

Treat these as semantic compatibility fixes with the Megatron-LM reference:

- compact image sentinel `-200` and no dataloader-side image expansion
- per-source packed collation boundaries and padding
- Nemotron pad-token parity
- auxiliary-media basename fallback for matching media resolution
- accepting literal `"<image>"` prose under the negative-sentinel image path

Treat these as parity/determinism or stream-order fixes:

- balanced greedy knapsack loop and `balanced_knapsack_delta`
- `task_encoder.packing` config plumbing
- removal of NeMo-only train dataset kwargs
- optional `pack_shuffle_seed`

The practical rule is: if a valid sample becomes different token IDs,
different media tensors, different masks, or different packed attention
boundaries, it is a functional mismatch. If the same valid samples appear in a
different order or co-pack with different neighbors, it is usually statistical
drift unless it changes dataset membership or packed-attention layout.
