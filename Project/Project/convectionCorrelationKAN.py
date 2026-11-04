import csv
import warnings
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from kan import KAN
from kan.utils import SYMBOLIC_LIB, create_dataset_from_data, ex_round


# Colab setup:
# %pip install -q git+https://github.com/KindXiaoming/pykan.git scikit-learn matplotlib numpy torch
# Restart runtime after install if `import kan` fails.

SEED = 42
CSV_PATH = "spoofed_convection_data.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_LOG_TARGET = True

SYMBOLIC_LIB_MULT_STYLE = ["0", "x", "log"]
SYMBOLIC_LIB_FALLBACK = ["0", "x", "x^2", "log"]


def get_supported_symbolic_lib(candidates: list[str]) -> list[str]:
    supported = [name for name in candidates if name in SYMBOLIC_LIB]
    if not supported:
        raise RuntimeError("No requested symbolic functions are supported by this pykan build.")
    return supported


torch.random.manual_seed(SEED)
np.random.seed(SEED)
torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def ensure_model_device(model: KAN, device: torch.device) -> KAN:
    model = cast(Any, model).to(str(device))
    cast(Any, model).device = str(device)
    return model


def load_data(csv_path: str):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    feature_names = [
        "Diameter D (m)",
        "Length-to-Diameter Ratio (L/D)",
        "Reynolds Number",
        "Prandtl Number",
        "Thermal Conductivity k (W/m*K)",
    ]
    target_name = "Convection Coefficient h (W/m^2K)"

    X_rows = []
    y_rows = []
    for row in rows:
        d = float(row["Diameter D (m)"])
        l = float(row["Length L (m)"])
        l_over_d = l / d
        X_rows.append([
            d,
            l_over_d,
            float(row["Reynolds Number"]),
            float(row["Prandtl Number"]),
            float(row["Thermal Conductivity k (W/m*K)"]),
        ])
        y_rows.append(float(row[target_name]))

    X_np = np.array(X_rows, dtype=np.float64)
    y_np = np.array(y_rows, dtype=np.float64).reshape(-1, 1)
    return X_np, y_np, feature_names, target_name


def fit_stage(
    model: KAN,
    dataset: dict[str, torch.Tensor],
    *,
    title: str,
    opt: str,
    steps: int,
    lr: float | None,
    lamb: float,
    update_grid: bool,
):
    print(title)
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "opt": opt,
        "steps": steps,
        "lamb": lamb,
        "lamb_l1": 1.0,
        "lamb_entropy": 2.0,
        "lamb_coef": 0.0,
        "lamb_coefdiff": 0.0,
        "reg_metric": "edge_forward_spline_n",
        "update_grid": update_grid,
    }
    if lr is not None and opt == "Adam":
        kwargs["lr"] = lr
        kwargs["batch"] = 256
    result = model.fit(**kwargs)
    model.get_act(dataset["train_input"])
    return result


def plot_histograms(X_np: np.ndarray, feature_names: list[str]):
    n_features = len(feature_names)
    n_cols = 3
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes_flat = np.atleast_1d(axes).ravel()
    for idx, ax in enumerate(axes_flat[:n_features]):
        ax.hist(X_np[:, idx], bins=40, color="steelblue", alpha=0.85)
        ax.set_title(feature_names[idx])
    for ax in axes_flat[n_features:]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_training_curves(stages: list[tuple[str, dict[str, list[float]]]]):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    offset = 0
    for name, result in stages:
        train = np.asarray(result["train_loss"], dtype=float)
        test = np.asarray(result["test_loss"], dtype=float)
        step_idx = np.arange(offset, offset + len(train))
        ax.plot(step_idx, train, label=f"{name} train")
        ax.plot(step_idx, test, linestyle="--", label=f"{name} test")
        offset += len(train)
    ax.set_title("KAN Training History")
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("MSE")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()


def try_symbolic(model: KAN, dataset: dict[str, torch.Tensor]):
    attempts = [
        ("mult-style", get_supported_symbolic_lib(SYMBOLIC_LIB_MULT_STYLE)),
        ("fallback", get_supported_symbolic_lib(SYMBOLIC_LIB_FALLBACK)),
    ]

    for attempt_name, lib in attempts:
        print(f"Symbolic attempt: {attempt_name}")
        model = ensure_model_device(model, DEVICE)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cast(Any, model).auto_symbolic(lib=lib, weight_simple=0.15, r2_threshold=0.0)

        symbolic_formula = cast(Any, model).symbolic_formula()
        if symbolic_formula is None:
            print("  symbolic_formula returned None")
            continue

        symbolic_formula = cast(Any, symbolic_formula)
        symbolic_expr = ex_round(symbolic_formula[0][0], 5)
        symbolic_expr_text = str(symbolic_expr)
        if "nan" in symbolic_expr_text.lower():
            print("  symbolic expression contains NaN")
            continue

        with torch.no_grad():
            y_sym = model(dataset["test_input"])
        if torch.isnan(y_sym).any() or not torch.isfinite(y_sym).all():
            print("  symbolic model produced non-finite predictions")
            continue

        r2_sym = r2_score(
            dataset["test_label"].detach().cpu().numpy(),
            y_sym.detach().cpu().numpy(),
        )
        print(f"  symbolic equation (scaled h): {symbolic_expr_text}")
        print(f"  symbolic R^2 (scaled): {r2_sym:.4f}")
        return symbolic_expr_text, float(r2_sym), y_sym

    raise RuntimeError("KAN symbolic conversion failed in all attempts.")


def main():
    X_np, y_np, feature_names, target_name = load_data(CSV_PATH)
    print(f"Samples: {X_np.shape[0]}")
    print(f"Features: {feature_names}")
    print(f"Target range ({target_name}): {y_np.min():.3f} to {y_np.max():.3f}")
    print(f"Using device: {DEVICE}")

    plot_histograms(X_np, feature_names)

    scaler_X = MinMaxScaler(feature_range=(0.05, 0.95))
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_np)
    y_model_np = np.log(y_np) if USE_LOG_TARGET else y_np
    y_scaled = scaler_y.fit_transform(y_model_np)

    X_t = torch.tensor(X_scaled, dtype=torch.float64, device=DEVICE)
    y_t = torch.tensor(y_scaled, dtype=torch.float64, device=DEVICE)

    np.random.seed(SEED)
    dataset = create_dataset_from_data(X_t, y_t, train_ratio=0.8, device=str(DEVICE))

    model = KAN(
        width=[[X_t.shape[1], 0], [1, 4], [1, 0]],
        mult_arity=2,
        grid=2,
        k=3,
        base_fun="identity",
        sparse_init=True,
        seed=SEED,
        device=str(DEVICE),
    )
    model = ensure_model_device(model, DEVICE)

    model.get_act(dataset["train_input"])
    model.plot()
    plt.title("Initial KAN")
    plt.tight_layout()
    plt.show()

    model.perturb(mode="all")
    model = ensure_model_device(model, DEVICE)
    model.get_act(dataset["train_input"])

    model.plot()
    plt.title("After Perturbation")
    plt.tight_layout()
    plt.show()

    stages: list[tuple[str, dict[str, list[float]]]] = []
    stages.append((
        "Adam warmup",
        fit_stage(
            model,
            dataset,
            title="\nStage 1: Adam warmup",
            opt="Adam",
            steps=250,
            lr=1e-3,
            lamb=1e-5,
            update_grid=False,
        ),
    ))
    stages.append((
        "LBFGS refine",
        fit_stage(
            model,
            dataset,
            title="Stage 2: LBFGS refine",
            opt="LBFGS",
            steps=80,
            lr=None,
            lamb=1e-6,
            update_grid=False,
        ),
    ))

    model = ensure_model_device(model, DEVICE)
    model.get_act(dataset["train_input"])
    plot_training_curves(stages)

    with torch.no_grad():
        y_pred_scaled = model(dataset["test_input"])
    r2_spline = r2_score(
        dataset["test_label"].detach().cpu().numpy(),
        y_pred_scaled.detach().cpu().numpy(),
    )
    print(f"Spline KAN R^2 (scaled): {r2_spline:.4f}")

    print("\nAttempting symbolic conversion...")
    model.get_act(dataset["train_input"])
    symbolic_expr_text, r2_symbolic, _ = try_symbolic(model, dataset)

    y_test_model = scaler_y.inverse_transform(dataset["test_label"].detach().cpu().numpy())
    y_pred_model = scaler_y.inverse_transform(y_pred_scaled.detach().cpu().numpy())
    if USE_LOG_TARGET:
        y_test_unscaled = np.exp(y_test_model)
        y_pred_unscaled = np.exp(y_pred_model)
    else:
        y_test_unscaled = y_test_model
        y_pred_unscaled = y_pred_model

    r2_h_space = r2_score(y_test_unscaled, y_pred_unscaled)
    print(f"Spline KAN R^2 (h-space): {r2_h_space:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    lo = float(min(y_test_unscaled.min(), y_pred_unscaled.min()))
    hi = float(max(y_test_unscaled.max(), y_pred_unscaled.max()))
    ax.scatter(y_test_unscaled, y_pred_unscaled, s=10, alpha=0.35)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_title(f"Parity (Spline R^2={r2_spline:.3f})")
    ax.set_xlabel(f"True {target_name}")
    ax.set_ylabel(f"Predicted {target_name}")

    ax = axes[1]
    residuals = (y_pred_unscaled - y_test_unscaled).ravel()
    ax.hist(residuals, bins=40, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.axvline(0.0, color="k", linestyle="--", lw=1)
    ax.set_title(f"Residuals (Symbolic R^2={r2_symbolic:.3f})")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")

    plt.tight_layout()
    plt.show()

    print("\nFinal symbolic equation (scaled model target):")
    print(symbolic_expr_text)
    if USE_LOG_TARGET:
        print("Interpreted form: h = exp(model_expression)")


if __name__ == "__main__":
    main()
