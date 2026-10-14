import json, pathlib, struct, subprocess, sys

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from train import CongestionNet, load, metrics  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_network_shape_and_range():
    m = CongestionNet()
    y = m(torch.rand(2, 5, 26, 40))
    assert y.shape == (2, 1, 26, 40)
    assert float(y.min()) >= 0 and float(y.max()) <= 1


def test_loader_reads_rust_binary_format(tmp_path):
    c, h, w, n = 5, 3, 4, 2
    x = np.random.rand(n, c, h, w).astype("<f4")
    y = np.random.rand(n, 1, h, w).astype("<f4")
    p = tmp_path / "d.bin"
    with open(p, "wb") as f:
        f.write((json.dumps({"c": c, "h": h, "w": w, "future": 8, "samples": n}) + "\n").encode())
        for i in range(n):
            f.write(x[i].tobytes() + y[i].tobytes())
    _, xr, yr = load(p)
    np.testing.assert_array_equal(xr, x)
    np.testing.assert_array_equal(yr, y)


def test_metrics_perfect_prediction():
    y = np.random.rand(4, 1, 5, 5).astype("f4")
    m = metrics(y, y, np.ones_like(y, dtype=bool))
    assert m["mse"] == 0 and abs(m["pearson_r"] - 1) < 1e-6 and m["hotspot_recall@5%"] == 1


def test_committed_weights_export_to_onnx_and_match_fixture(tmp_path):
    out = tmp_path / "m.onnx"
    subprocess.run([sys.executable, str(ROOT / "ml/export_onnx.py"), "--weights", str(ROOT / "model/congestion.json"),
                    "--out", str(out)], check=True, cwd=ROOT / "ml")
    fx = json.loads((ROOT / "results/parity_fixture.json").read_text())
    sess = ort.InferenceSession(str(out))
    for xi, yi in zip(fx["inputs"], fx["outputs"]):
        x = np.asarray(xi, dtype=np.float32).reshape(1, 5, fx["h"], fx["w"])
        np.testing.assert_allclose(sess.run(None, {"state": x})[0].ravel(), yi, atol=1e-5)
