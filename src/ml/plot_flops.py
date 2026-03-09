from thop import profile, clever_format
import torch
import matplotlib.pyplot as plt
from src.channels.channel_est.ml_channel import VAE
from src.settings.ml import device


def compute_flops(model, input_shape):
    model = model.to(device).eval()
    dummy_input = torch.randn(1, *input_shape).to(device)
    flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
    flops_fmt = clever_format([flops], "%.3f")[0]
    print(f"{'Input shape':<20}: {input_shape}")
    print(f"{'FLOPs':<20}: {flops_fmt}  ({flops:,.0f})")
    return flops


def count_params(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{'Parameters':<20}: {params / 1e6:.3f} M  ({params:,})")
    return params


# ── Input shapes to sweep ─────────────────────────────────────────────────────
input_shapes = {
    (2, 32, 64): r"$1 \times 2$",
    (2, 64, 64): r"$2 \times 2$",
    (2, 128, 64): r"$2 \times 4$",
    (2, 256, 64): r"$4 \times 4$",
}

print("=" * 55)
print("VAE FLOPs vs Input Shape")
print("=" * 55)

vae_flops, vae_params = [], []

for shape, label in input_shapes.items():
    c, h, w = shape
    mu_h = torch.zeros(1, c * h * w).to(device)
    std_h = torch.ones(1, c * h * w).to(device)

    model = VAE(input_shape=shape, mu_h=mu_h, std_h=std_h)

    flops = compute_flops(model, shape)
    params = count_params(model)

    vae_flops.append(flops / 1e9)  # GFLOPs
    vae_params.append(params / 1e6)  # M params


# ── Plot ──────────────────────────────────────────────────────────────────────
labels = list(input_shapes.values())
x = list(range(len(labels)))

fig, ax1 = plt.subplots(figsize=(6, 3))
ax2 = ax1.twinx()

(l1,) = ax1.plot(
    x,
    vae_flops,
    marker="o",
    color="tab:blue",
    linewidth=1.8,
    markersize=6,
    label="FLOPs",
)
(l2,) = ax2.plot(
    x,
    vae_params,
    marker="s",
    color="tab:orange",
    linestyle="--",
    linewidth=1.8,
    markersize=6,
    label="Params (M)",
)

ax1.set_xlabel(
    r"Antenna setting $K_{\rm ut,ant} \times K_{\rm bs,ant}$",
    fontsize=12,
)
ax1.set_ylabel("FLOPs", fontsize=12, color="tab:blue")
ax2.set_ylabel("Parameters (M)", fontsize=12, color="tab:orange")
ax1.tick_params(axis="y", labelcolor="tab:blue", labelsize=11)
ax2.tick_params(axis="y", labelcolor="tab:orange", labelsize=11)
ax1.tick_params(axis="x", labelsize=11)
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_title(
    r"VAE Complexity vs. Antenna Configuration",
    fontsize=13,
)
ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
ax1.legend(handles=[l1, l2], fontsize=10, loc="upper left")

plt.tight_layout()
plt.savefig("vae_complexity.png", dpi=300)
plt.show()
