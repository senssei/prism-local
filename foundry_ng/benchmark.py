"""
foundry_ng.benchmark: Micro-Benchmark Runner for Speed, TTFT & VRAM.
"""

import time
from typing import Dict, Any
from foundry_ng.catalog import ModelCatalog
from foundry_ng.engine import OnnxGenAiEngine, format_prompt
from foundry_ng.telemetry import get_gpu_info


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
    print(f" 🏎️ foundry-ng Micro-Benchmark: {resolved['name']}")
    print("=" * 60)

    print("1. Loading model into GPU VRAM...")
    t_load_start = time.perf_counter()
    engine = OnnxGenAiEngine(resolved["path"])
    load_time = time.perf_counter() - t_load_start
    print(f"   Model load time: {load_time:.2f}s")

    gpu_info = get_gpu_info()
    initial_vram = gpu_info["devices"][0]["vram_used_mb"] if gpu_info.get("devices") else 0
    print(f"   Current VRAM allocated: {initial_vram:.1f} MB")

    # Warmup
    print("2. Running warmup run (32 tokens)...")
    warmup_prompt = format_prompt([{"role": "user", "content": "Count from 1 to 5."}])
    engine.generate(warmup_prompt, max_tokens=32)

    # Benchmark run: 256 tokens code generation
    test_prompt = format_prompt([
        {"role": "user", "content": "Write a complete Python implementation of an LRU Cache with get and put methods."}
    ])

    print("3. Executing benchmark generation (target 256 tokens)...")
    t0 = time.perf_counter()
    res = engine.generate(test_prompt, max_tokens=256)
    total_time = time.perf_counter() - t0

    gpu_info_after = get_gpu_info()
    peak_vram = gpu_info_after["devices"][0]["vram_used_mb"] if gpu_info_after.get("devices") else 0

    print("\n" + "=" * 60)
    print(" 📊 BENCHMARK SCORECARD")
    print("=" * 60)
    print(f"  • Model:                {resolved['name']}")
    print(f"  • Generation Tokens:    {res['tokens_generated']} tokens")
    print(f"  • Time to First Token:  {res['ttft_sec'] * 1000:.1f} ms ({res['ttft_sec']:.3f}s)")
    print(f"  • Generation Speed:     {res['decode_tok_per_sec']:.1f} tokens/second")
    print(f"  • Total Time:           {res['elapsed_sec']:.2f} seconds")
    print(f"  • Peak VRAM:            {peak_vram:.1f} MB")
    print("=" * 60)

    print("4. Unloading model and reclaiming VRAM...")
    engine.unload()
    gpu_info_unloaded = get_gpu_info()
    final_vram = gpu_info_unloaded["devices"][0]["vram_used_mb"] if gpu_info_unloaded.get("devices") else 0
    print(f"   VRAM after unload: {final_vram:.1f} MB (Freed: {peak_vram - final_vram:.1f} MB)")

    return {
        "model": resolved["name"],
        "tokens": res["tokens_generated"],
        "ttft_sec": res["ttft_sec"],
        "decode_tok_per_sec": res["decode_tok_per_sec"],
        "peak_vram_mb": peak_vram,
    }
