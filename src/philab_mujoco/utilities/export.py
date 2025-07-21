import onnx
import numpy as np
import torch
from torch import nn


def params_to_pt(params, input_dim, activation=nn.ReLU):
    layers = []
    param_layers = [k for k in params.keys() if k.startswith("hidden_")]
    param_layers.sort(key=lambda x: int(x.split("_")[1]))
    prev_dim = input_dim

    for i, layer_name in enumerate(param_layers):
        bias = np.array(params[layer_name]["bias"])
        out_features = bias.shape[0]
        kernel = np.array(params[layer_name]["kernel"]).reshape(out_features, prev_dim)

        linear = nn.Linear(prev_dim, out_features)
        linear.weight.data = torch.from_numpy(kernel).float()
        linear.bias.data = torch.from_numpy(bias).float()
        layers.append(linear)

        if i < len(param_layers) - 1:
            layers.append(activation())
        prev_dim = out_features

    return nn.Sequential(*layers)


def params_to_onnx(params, input_dim, onnx_path="mlp.onnx"):
    mlp = params_to_pt(params, input_dim)
    mlp.eval()
    dummy_input = torch.randn(1, input_dim)

    torch.onnx.export(
        mlp,
        dummy_input,
        onnx_path,
        export_params=True,
        input_names=["input"],
        output_names=["output"]
    )

    print(f"Exported to {onnx_path}")


def print_tree_with_numpy_sizes(d, level=0):
    indent = "    " * level
    for key, value in d.items():
        if isinstance(value, dict):
            print(f"{indent}|-{key}")
            print_tree_with_numpy_sizes(value, level + 1)
        elif isinstance(value, np.ndarray):
            print(f"{indent}|-{key} (size = {value.size})")
        else:
            print(f"{indent}|-{key} (type = {type(value).__name__})")