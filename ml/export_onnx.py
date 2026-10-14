"""Rebuild congestion.onnx from the committed JSON weights (the canonical trained model).

CI uses this so the deployed ONNX file is byte-for-byte derived from reviewed, text-diffable weights.
"""
import argparse, json, pathlib

import torch

from train import CongestionNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="../model/congestion.json")
    ap.add_argument("--out", default="../model/congestion.onnx")
    ap.add_argument("--h", type=int, default=26)
    ap.add_argument("--w", type=int, default=40)
    a = ap.parse_args()
    layers = json.loads(pathlib.Path(a.weights).read_text())["layers"]
    model = CongestionNet(cin=layers[0]["cin"], width=layers[0]["cout"], dilations=tuple(l["dilation"] for l in layers[:-1]))
    convs = list(model.body) + [model.head]
    with torch.no_grad():
        for conv, l in zip(convs, layers):
            conv.weight.copy_(torch.tensor(l["weight"]).reshape(conv.weight.shape))
            conv.bias.copy_(torch.tensor(l["bias"]))
    model.eval()
    torch.onnx.export(model, torch.zeros(1, layers[0]["cin"], a.h, a.w), a.out, input_names=["state"],
                      output_names=["congestion"], opset_version=17, dynamo=False)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
