"""Train the fleet congestion predictor ("DeepFleet-lite").

Input : 5-channel fleet-state tensor from the Rust simulator (shelves, stations, robots, targets,
        8-step shortest-path intent).
Target: per-cell mean robot occupancy over the next 8 ticks.

Compares three predictors on a held-out split:
  * reactive heuristic  - blurred current occupancy (what the planner's "density" mode uses)
  * scikit-learn        - HistGradientBoostingRegressor on 3x3 neighbourhood features
  * PyTorch CNN         - dilated fully-convolutional network (exported to ONNX + JSON)
"""
import argparse, json, pathlib, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingRegressor


def load(path):
    with open(path, "rb") as f:
        header = json.loads(f.readline())
        raw = np.frombuffer(f.read(), dtype="<f4")
    c, h, w = header["c"], header["h"], header["w"]
    per = (c + 1) * h * w
    n = raw.size // per
    raw = raw[: n * per].reshape(n, per)
    x = raw[:, : c * h * w].reshape(n, c, h, w)
    y = raw[:, c * h * w :].reshape(n, 1, h, w)
    return header, x.copy(), y.copy()


class CongestionNet(nn.Module):
    """Dilated FCN: receptive field 31x31 with ~7k parameters."""

    def __init__(self, cin=5, width=16, dilations=(1, 2, 4, 8)):
        super().__init__()
        layers, c = [], cin
        for d in dilations:
            layers.append(nn.Conv2d(c, width, 3, padding=d, dilation=d))
            c = width
        self.body = nn.ModuleList(layers)
        self.head = nn.Conv2d(width, 1, 1)
        nn.init.constant_(self.head.bias, -3.0)  # start near the base rate (~5% occupancy)
        self.dilations = dilations

    def forward(self, x):
        for conv in self.body:
            x = F.relu(conv(x))
        return torch.sigmoid(self.head(x))


def blur(occ, k=5):
    t = torch.from_numpy(occ)
    ker = torch.ones(1, 1, k, k) / 6.0
    return torch.clamp(F.conv2d(t, ker, padding=k // 2), 0, 1).numpy()


def metrics(pred, y, free):
    p, t = pred[free], y[free]
    mse = float(np.mean((p - t) ** 2))
    mae = float(np.mean(np.abs(p - t)))
    corr = float(np.corrcoef(p, t)[0, 1])
    # hotspot recall: of the top-5% truly busiest cells, how many are in the predicted top-5%
    k = max(1, int(0.05 * t.size))
    top_t = set(np.argpartition(-t, k)[:k].tolist())
    top_p = set(np.argpartition(-p, k)[:k].tolist())
    return {"mse": mse, "mae": mae, "pearson_r": corr, "hotspot_recall@5%": len(top_t & top_p) / k}


def patches(x, idx_n, idx_y, idx_x):
    n, c, h, w = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    feats = [xp[idx_n, :, idx_y + dy, idx_x + dx] for dy in range(3) for dx in range(3)]
    return np.concatenate(feats, axis=1)


def export(model, out, h, w):
    out.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.zeros(1, 5, h, w)
    torch.onnx.export(model, dummy, out / "congestion.onnx", input_names=["state"], output_names=["congestion"],
                      opset_version=17, dynamo=False)
    layers = []
    for conv, d in zip(model.body, model.dilations):
        layers.append(dict(cin=conv.in_channels, cout=conv.out_channels, k=3, dilation=d, act="relu",
                           weight=conv.weight.detach().flatten().tolist(), bias=conv.bias.detach().tolist()))
    hd = model.head
    layers.append(dict(cin=hd.in_channels, cout=1, k=1, dilation=1, act="sigmoid",
                       weight=hd.weight.detach().flatten().tolist(), bias=hd.bias.detach().tolist()))
    (out / "congestion.json").write_text(json.dumps({"layers": layers}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.bin")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    header, x, y = load(a.data)
    n, c, h, w = x.shape
    split = int(0.8 * n)  # samples are written simulation by simulation: tail = unseen simulations
    xtr, ytr, xva, yva = x[:split], y[:split], x[split:], y[split:]
    free_va = (xva[:, 0:1] == 0)
    print(f"data: {n} samples {c}x{h}x{w}; train {split} / val {n - split}")
    report = {"samples": n, "grid": [h, w], "future_ticks": header["future"]}

    # 1) reactive heuristic
    report["reactive_blur"] = metrics(blur(xva[:, 2:3]), yva, free_va)

    # 2) scikit-learn gradient boosting on local patches
    rng = np.random.default_rng(a.seed)
    m = 120_000
    tn, ty, tx = rng.integers(0, split, m), rng.integers(0, h, m), rng.integers(0, w, m)
    t0 = time.time()
    gbr = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=a.seed)
    gbr.fit(patches(xtr, tn, ty, tx), ytr[tn, 0, ty, tx])
    vn, vy, vx = np.meshgrid(np.arange(n - split), np.arange(h), np.arange(w), indexing="ij")
    pv = gbr.predict(patches(xva, vn.ravel(), vy.ravel(), vx.ravel())).reshape(n - split, 1, h, w)
    report["sklearn_hgb_3x3"] = metrics(np.clip(pv, 0, 1), yva, free_va)
    report["sklearn_fit_s"] = round(time.time() - t0, 1)

    # 3) PyTorch dilated CNN
    model = CongestionNet()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    Xtr, Ytr = torch.from_numpy(xtr), torch.from_numpy(ytr)
    Xva, Yva = torch.from_numpy(xva), torch.from_numpy(yva)
    best, best_state = 1e9, None
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(split)
        for i in range(0, split, 32):
            b = perm[i : i + 32]
            xb, yb = Xtr[b], Ytr[b]
            if torch.rand(()) < 0.5:  # mirror augmentation: the layout is left/right symmetric
                xb, yb = xb.flip(-1), yb.flip(-1)
            loss = F.mse_loss(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = F.mse_loss(model(Xva), Yva).item()
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep:2d} val_mse {vl:.5f}")
    model.load_state_dict(best_state)
    with torch.no_grad():
        pc = model(Xva).numpy()
    report["cnn"] = metrics(pc, yva, free_va)
    report["cnn_params"] = sum(p.numel() for p in model.parameters())
    report["cnn_train_s"] = round(time.time() - t0, 1)

    out = pathlib.Path(a.out)
    export(model, out, h, w)

    # parity fixture: ONNX Runtime outputs for a few validation states (checked against Rust in CI)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out / "congestion.onnx"))
    xs = xva[:4]
    ys = sess.run(None, {"state": xs[:1]})[0]
    with torch.no_grad():
        report["onnx_vs_torch_max_abs"] = float(np.abs(ys - model(torch.from_numpy(xs[:1])).numpy()).max())
    fixture = {"h": h, "w": w, "inputs": [xs[i].ravel().tolist() for i in range(2)],
               "outputs": [sess.run(None, {"state": xs[i : i + 1]})[0].ravel().tolist() for i in range(2)]}
    (out / "parity_fixture.json").write_text(json.dumps(fixture))
    (out / "train_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()}, indent=2))


if __name__ == "__main__":
    main()
