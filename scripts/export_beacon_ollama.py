from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_BASE_MODEL = "/kaggle/input/models/google/gemma-4/transformers/gemma-4-e2b-it/1"
DEFAULT_ADAPTER = "/kaggle/input/beacon-tool-dpo-from-cpt-fullprompt-ckpt50-adapter"
DEFAULT_OUT_DIR = "/kaggle/working/beacon_ollama_export"
DEFAULT_GGUF_NAME = "beacon-gemma4-e2b-current-best-q4_k_m.gguf"
LLAMA_CPP_DIR = Path("/tmp/beacon_llama_cpp")


def patch_tokenizer_save_pretrained(tokenizer_obj: Any, source_dir: Path) -> Any:
    def save_pretrained(save_directory: str | Path, *args: Any, **kwargs: Any) -> tuple[str, ...]:
        dest = Path(save_directory)
        if str(dest).startswith("/kaggle/input"):
            dest = source_dir
        dest.mkdir(parents=True, exist_ok=True)
        copied: list[str] = []
        for name in [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.jinja",
            "processor_config.json",
            "preprocessor_config.json",
        ]:
            source = source_dir / name
            if source.exists():
                target = dest / name
                if source.resolve() != target.resolve():
                    shutil.copy2(source, target)
                copied.append(str(target))
        if not copied:
            raise RuntimeError(f"No tokenizer files found to copy from {source_dir}")
        return tuple(copied)

    tokenizer_obj.save_pretrained = save_pretrained
    return tokenizer_obj


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_command(command: list[str], log_path: Path, cwd: Path | None = None) -> None:
    result = subprocess.run(command, cwd=str(cwd) if cwd else None, text=True, capture_output=True)
    write_json(
        log_path,
        {
            "command": command,
            "cwd": str(cwd) if cwd else None,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-10000:],
            "stderr_tail": result.stderr[-10000:],
        },
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}")


def direct_llama_cpp_export(merged_dir: Path, out_dir: Path, quantization: str, gguf_name: str) -> Path:
    run_command([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision"], out_dir / "torchvision_uninstall_log.json")
    if not LLAMA_CPP_DIR.exists():
        run_command(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", str(LLAMA_CPP_DIR)], out_dir / "llama_cpp_clone_log.json")
    requirements = LLAMA_CPP_DIR / "requirements.txt"
    if requirements.exists():
        run_command([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", str(requirements)], out_dir / "llama_cpp_requirements_log.json", cwd=LLAMA_CPP_DIR)
    run_command(["cmake", "-B", "build", "-DGGML_CUDA=OFF", "-DLLAMA_CURL=OFF"], out_dir / "llama_cpp_cmake_configure_log.json", cwd=LLAMA_CPP_DIR)
    run_command(["cmake", "--build", "build", "--config", "Release", "-j", "2"], out_dir / "llama_cpp_cmake_build_log.json", cwd=LLAMA_CPP_DIR)
    converter = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    quantizer_candidates = [
        LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize",
        LLAMA_CPP_DIR / "build" / "bin" / "quantize",
        LLAMA_CPP_DIR / "llama-quantize",
        LLAMA_CPP_DIR / "quantize",
    ]
    quantizer = next((path for path in quantizer_candidates if path.exists()), None)
    if not converter.exists():
        raise RuntimeError(f"Missing llama.cpp converter: {converter}")
    if quantizer is None:
        raise RuntimeError(f"Missing llama.cpp quantizer in candidates: {[str(path) for path in quantizer_candidates]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    f16_path = out_dir / "beacon-gemma4-current-best-f16.gguf"
    q4_path = out_dir / gguf_name
    run_command([sys.executable, str(converter), str(merged_dir), "--outfile", str(f16_path), "--outtype", "f16"], out_dir / "convert_hf_to_gguf_log.json", cwd=LLAMA_CPP_DIR)
    run_command([str(quantizer), str(f16_path), str(q4_path), quantization.upper()], out_dir / "llama_quantize_log.json", cwd=LLAMA_CPP_DIR)
    if not q4_path.exists():
        raise RuntimeError(f"Direct llama.cpp export did not create {q4_path}")
    try:
        f16_path.unlink()
    except FileNotFoundError:
        pass
    return q4_path


def export_gguf(args: argparse.Namespace) -> None:
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        from peft import PeftModel
        from transformers import AutoProcessor, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing export dependencies. On Kaggle, install pinned Unsloth/PEFT first, for example:\n"
            "python -m pip install --no-cache-dir unsloth==2026.5.2 unsloth_zoo==2026.5.1 "
            "transformers==5.5.0 peft==0.19.1 accelerate==1.13.0 bitsandbytes==0.49.2 sentencepiece"
        ) from exc

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
    model = PeftModel.from_pretrained(model, args.adapter)

    merged_dir = out_dir / "merged_hf"
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    try:
        gguf_tokenizer = AutoProcessor.from_pretrained(str(merged_dir), trust_remote_code=True)
    except Exception:
        gguf_tokenizer = AutoTokenizer.from_pretrained(str(merged_dir), trust_remote_code=True)
    gguf_tokenizer = patch_tokenizer_save_pretrained(gguf_tokenizer, merged_dir)
    del gguf_tokenizer
    del model
    del merged_model
    gc.collect()
    direct_llama_cpp_export(merged_dir, out_dir / "gguf_direct", args.quantization, args.expected_gguf_name)
    export_backend = "direct_llama_cpp_fresh_clone"
    write_json(
        out_dir / "export_manifest.json",
        {
            "base_model": args.base_model,
            "adapter": args.adapter,
            "merged_hf": str(merged_dir),
            "quantization": args.quantization,
            "export_backend": export_backend,
            "expected_gguf_dir": str(out_dir),
            "ollama_import_hint": "Copy the produced .gguf beside ollama/Modelfile, update FROM if needed, then run: ollama create beacon-gemma4 -f ollama/Modelfile",
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge Beacon Gemma adapter and export GGUF for Ollama/llama.cpp.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--quantization", default="q4_k_m")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--expected-gguf-name", default=DEFAULT_GGUF_NAME)
    parser.set_defaults(func=export_gguf)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
