"""
Real-hardware smoke tests. Skipped unless onnxruntime_genai, an NVIDIA GPU (NVML) and a CUDA ONNX
model are all available, so they never run on CI. To run locally:

    PRISM_PYTHON=/path/to/venv/bin/python3 PRISM_MODEL_DIRS=/path/to/models \
        PYTHONPATH=. "$PRISM_PYTHON" -m unittest tests.test_prism_gpu_integration -v
"""

import http.client
import json
import threading
import unittest

from prism.catalog import ModelCatalog
from prism.engine import OG_AVAILABLE, OnnxGenAiEngine
from prism.server import ActiveEngineManager, create_server
from prism.telemetry import get_gpu_info


def _find_cuda_model():
    if not OG_AVAILABLE or not get_gpu_info().get("available"):
        return None
    for m in ModelCatalog().discover_onnx_models():
        if m["device"].startswith("CUDA"):
            return m
    return None


MODEL = _find_cuda_model()


@unittest.skipUnless(MODEL, "needs onnxruntime_genai + NVIDIA GPU + a CUDA ONNX model (see module docstring)")
class TestRealCudaEngine(unittest.TestCase):
    def test_engine_generates_and_reports_metrics(self):
        engine = OnnxGenAiEngine(MODEL["path"])
        try:
            prompt = "<|user|>\nSay hello.<|end|>\n<|assistant|>\n" if MODEL["template"] == "phi4" else "Say hello."
            self.assertGreater(engine.count_tokens(prompt), 0)
            result = engine.generate(prompt, max_tokens=8, temperature=0.0)
            self.assertGreater(result["tokens_generated"], 0)
            self.assertTrue(result["text"].strip())
            self.assertIn(result["finish_reason"], ("stop", "length"))
            self.assertGreater(result["decode_tok_per_sec"], 0)
        finally:
            engine.unload()

    def test_server_round_trip_with_real_engine(self):
        server = create_server(port=0, host="127.0.0.1", manager=ActiveEngineManager())
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        self.addCleanup(server.manager.unload)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        body = json.dumps({"model": MODEL["id"], "max_tokens": 6, "temperature": 0.0,
                           "messages": [{"role": "user", "content": "Count from 1 to 200, separated by commas."}]})
        conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=120)
        conn.request("POST", "/v1/chat/completions", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read())
        self.assertEqual(resp.status, 200, data)
        self.assertTrue(data["choices"][0]["message"]["content"].strip())
        self.assertGreater(data["usage"]["prompt_tokens"], 0)
        self.assertEqual(data["choices"][0]["finish_reason"], "length")  # a 200-number count cannot fit in 6 tokens


if __name__ == "__main__":
    unittest.main()
