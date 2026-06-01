"""Run a single prompt through a vLLM Llama-3.1-8B engine in prefill mode (eager),
capture inputs to every module on the requested decoder layers, and save the
final logits.

BF16 (default):
    uv run --extra vllm python my_script/vllm_forward.py

MXFP8 (pass ``--mxfp8`` and point ``--model`` at an MXFP8 checkpoint produced
by ``my_script/convert_hf_bf16_ckpt_to_mxfp8.py``):
    uv run --extra vllm python my_script/vllm_forward.py \\
        --mxfp8 --model /path/to/llama3.1-8b-instruct-mxfp8

The converter writes a ``config.json`` with
``quantization_config = {"quant_algo": "MXFP8", "quant_method": "modelopt"}``
which vLLM auto-detects via the ModelOpt quantization path; no extra kernel
flag is needed on the ``LLM(...)`` call. The MXFP8 ckpt dir may not include
tokenizer files — pass ``--tokenizer meta-llama/Llama-3.1-8B-Instruct`` to
pin tokenization to the canonical BF16 HF id.
"""

import argparse
import os

# Required so apply_model() can ship our hook-installer closure to the worker
# process via pickle (the default msgpack encoder rejects functions).
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

# Parse --batch-invariant *before* importing vllm so we can flip VLLM_BATCH_INVARIANT
# in the environment that the worker process will inherit.
_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--batch-invariant", action="store_true")
_pre_args, _ = _pre_parser.parse_known_args()
if _pre_args.batch_invariant:
    os.environ["VLLM_BATCH_INVARIANT"] = "1"

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


DEFAULT_PROMPT = "The quick brown fox jumps over the lazy dog."
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def default_output(batch_invariant: bool, mxfp8: bool) -> str:
    parts = ["vllm_capture"]
    if mxfp8:
        parts.append("mxfp8")
    if batch_invariant:
        parts.append("bi")
    name = "_".join(parts) + ".pt"
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="Model path or HF id. For --mxfp8, pass the MXFP8 "
                        "ckpt path produced by convert_hf_bf16_ckpt_to_mxfp8.py.")
    p.add_argument("--tokenizer", default=None,
                   help="Tokenizer source (HF id or path). Defaults to "
                        "--model; useful when the MXFP8 ckpt dir does not "
                        "bundle tokenizer files.")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--output", default=None)
    p.add_argument("--batch-invariant", action="store_true")
    p.add_argument("--mxfp8", action="store_true",
                   help="Run the model in MXFP8 precision. Requires --model "
                        "to point at an MXFP8-quantized ckpt (vLLM detects "
                        "the quantization from the ckpt's quantization_config).")
    p.add_argument("--capture-layers", default="0",
                   help="Comma-separated 0-indexed decoder layer numbers to capture "
                        "per-module input tensors for (default: 0).")
    args = p.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    if args.output is None:
        args.output = default_output(args.batch_invariant, args.mxfp8)
    args.capture_layers = [int(x) for x in args.capture_layers.split(",") if x.strip()]
    return args


def main():
    args = parse_args()
    print(f"[vllm] precision={'mxfp8' if args.mxfp8 else 'bf16'} "
          f"batch_invariant={args.batch_invariant} "
          f"(VLLM_BATCH_INVARIANT={os.environ.get('VLLM_BATCH_INVARIANT', '0')})")
    print(f"[vllm] model:     {args.model}")
    print(f"[vllm] tokenizer: {args.tokenizer}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    token_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    print(f"[vllm] prompt: {args.prompt!r}")
    print(f"[vllm] token ids ({len(token_ids)}): {token_ids}")

    # vLLM auto-detects MXFP8 via the ckpt's `quantization_config`; no extra
    # kwarg is needed. We keep activations in bf16 in both paths.
    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        enforce_eager=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        seed=0,
    )

    output_path = args.output
    prompt_text = args.prompt
    capture_layers = args.capture_layers

    def setup_hooks(model):
        """Register hooks on every decoder layer output, the LM head, and on
        all submodules of each layer in `capture_layers`.

        Runs inside the vLLM worker process; stash state on the model itself so
        a follow-up apply_model() call can save it after generate() finishes.
        """
        captured = {
            "module_inputs_by_layer": {idx: {} for idx in capture_layers},
            "layer_outputs": {},
            "logits": None,
        }
        handles = []

        # vLLM's LlamaDecoderLayer returns (hidden_states, residual). The actual
        # residual-stream value at the layer boundary is hidden_states + residual.
        def make_layer_output_hook(idx):
            def hook(module, args_, output_):
                if isinstance(output_, tuple) and len(output_) == 2:
                    h, r = output_
                    if isinstance(h, torch.Tensor) and isinstance(r, torch.Tensor):
                        rs = (h + r).detach().to(torch.float32).cpu().clone()
                        captured["layer_outputs"][idx] = rs
            return hook

        for i, layer in enumerate(model.model.layers):
            handles.append(layer.register_forward_hook(make_layer_output_hook(i)))

        def make_input_hook(layer_idx, name):
            bucket = captured["module_inputs_by_layer"][layer_idx]
            def hook(module, args_, output_):
                saved = []
                for a in args_:
                    if isinstance(a, torch.Tensor):
                        saved.append(a.detach().to(torch.float32).cpu().clone())
                    else:
                        saved.append(a)
                # Only record the first call (prefill); ignore any later calls.
                if name not in bucket:
                    bucket[name] = saved
            return hook

        for layer_idx in capture_layers:
            layer = model.model.layers[layer_idx]
            for name, sub in layer.named_modules():
                qual = name if name else "<layer>"
                handles.append(sub.register_forward_hook(make_input_hook(layer_idx, qual)))

        def logits_hook(module, args_, output_):
            # LogitsProcessor.forward returns the final logits tensor.
            if captured["logits"] is None and isinstance(output_, torch.Tensor):
                captured["logits"] = output_.detach().to(torch.float32).cpu().clone()
        handles.append(model.logits_processor.register_forward_hook(logits_hook))

        model._capture_state = (captured, handles)
        return None

    def save_and_cleanup(model):
        captured, handles = model._capture_state
        for h in handles:
            h.remove()
        payload = {
            "prompt": prompt_text,
            "token_ids": token_ids,
            # Legacy alias for backward compat with older compare.py.
            "first_layer_inputs": captured["module_inputs_by_layer"].get(0, {}),
            "module_inputs_by_layer": captured["module_inputs_by_layer"],
            "layer_outputs": captured["layer_outputs"],
            "logits": captured["logits"],
        }
        torch.save(payload, output_path)
        return output_path

    llm.apply_model(setup_hooks)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        prompt_logprobs=0,
        seed=0,
    )
    outputs = llm.generate(
        [{"prompt_token_ids": token_ids}],
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    print(f"[vllm] generated {len(outputs[0].outputs[0].token_ids)} new token(s)")

    saved_paths = llm.apply_model(save_and_cleanup)
    print(f"[vllm] saved capture to {saved_paths[0]}")


if __name__ == "__main__":
    main()
