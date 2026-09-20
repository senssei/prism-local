"""
prism.benchmark: Micro-Benchmark Runner for Speed, TTFT & VRAM.
"""

import time
from typing import Dict, Any
from prism.catalog import ModelCatalog
from prism.engine import OnnxGenAiEngine, format_prompt
from prism.telemetry import get_gpu_info


def _vram_used() -> float:
    devices = get_gpu_info(max_age=0).get("devices")
    return devices[0]["vram_used_mb"] if devices else 0.0


def run_benchmark(model_id_or_alias: str) -> Dict[str, Any]:
    catalog = ModelCatalog()
    resolved = catalog.resolve_model(model_id_or_alias)
    if not resolved:
        print(f"❌ Model '{model_id_or_alias}' not found.")
        return {}

    if resolved.get("backend") == "ollama":
        print(f"ℹ️ Model is an Ollama model. Running quick probe...")
        # For Ollama, we can run a simple check or recommend running BenchRig
        return {"model": model_id_or_alias, "backend": "ollama"}

    print("=" * 60)
    print(f" 🏎️  prism Micro-Benchmark: {resolved['name']}")
    print("=" * 60)

    baseline_vram = _vram_used()
    print("1. Loading model...")
    t_load_start = time.perf_counter()
    engine = OnnxGenAiEngine(resolved["path"])
    load_time = time.perf_counter() - t_load_start
    print(f"   Model load time: {load_time:.2f}s")
    print(f"   Execution provider: {engine.device.upper()}")
    if engine.fallback_reason:
        print(f"   ⚠️  NOT running on the GPU: {engine.fallback_reason}")
    print(f"   VRAM added by load: {_vram_used() - baseline_vram:+.1f} MB")

    # Warmup
    print("2. Running warmup run (32 tokens)...")
    warmup_prompt = format_prompt([{"role": "user", "content": "Count from 1 to 5."}], resolved.get("template"))
    engine.generate(warmup_prompt, max_tokens=32)

    # Benchmark run: 256 tokens code generation
    test_prompt = format_prompt([
        {"role": "user", "content": "Write a complete Python implementation of an LRU Cache with get and put methods."}
    ], resolved.get("template"))

    print("3. Executing benchmark generation (target 256 tokens)...")
    t0 = time.perf_counter()
    res = engine.generate(test_prompt, max_tokens=256)
    total_time = time.perf_counter() - t0

    peak_vram = _vram_used()

    print("\n" + "=" * 60)
    print(" 📊 BENCHMARK SCORECARD")
    print("=" * 60)
    print(f"  • Model:                {resolved['name']}")
    print(f"  • Execution Provider:   {engine.device.upper()}")
    print(f"  • Generation Tokens:    {res['tokens_generated']} tokens")
    print(f"  • Time to First Token:  {res['ttft_sec'] * 1000:.1f} ms ({res['ttft_sec']:.3f}s)")
    print(f"  • Generation Speed:     {res['decode_tok_per_sec']:.1f} tokens/second")
    print(f"  • Total Time:           {res['elapsed_sec']:.2f} seconds")
    print(f"  • VRAM used by model:   {peak_vram - baseline_vram:+.1f} MB (device total {peak_vram:.1f} MB; other apps share it)")
    print("=" * 60)

    print("4. Unloading model and reclaiming VRAM...")
    engine.unload()
    final_vram = _vram_used()
    print(f"   VRAM after unload: {final_vram:.1f} MB (freed {peak_vram - final_vram:.1f} MB; "
          f"{final_vram - baseline_vram:+.1f} MB vs. before load)")

    return {
        "model": resolved["name"],
        "tokens": res["tokens_generated"],
        "ttft_sec": res["ttft_sec"],
        "decode_tok_per_sec": res["decode_tok_per_sec"],
        "device": engine.device,
        "vram_used_by_model_mb": round(peak_vram - baseline_vram, 1),
        "vram_after_unload_vs_baseline_mb": round(final_vram - baseline_vram, 1),
    }
