#!/bin/python

import torch
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ---------- polynomial ----------
def poly(x, c):
    return (((c[0]*x + c[1])*x + c[2])*x + c[3])*x + c[4]

def dpoly(x, c):
    return (4*c[0]*x**3 + 3*c[1]*x**2 + 2*c[2]*x + c[3])

# ---------- Newton ----------
def newton(c, x0, iters=20, tol=1e-10):
    x = x0
    for step in range(1, iters+1):
        fx  = poly(x, c)
        dfx = dpoly(x, c)
        if abs(dfx) < 1e-12:
            break
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            return x_new, step
        x = x_new
    return x, iters

# ---------- Generate dataset ----------
def generate(N=5000):
    X = np.zeros((N, 5), dtype=np.float32)
    Y = np.zeros((N, 1), dtype=np.float32)
    for i in range(N):
        roots = np.random.uniform(-5, 5, 4)
        c     = np.poly(roots)
        X[i]  = c
        Y[i]  = roots[np.argmin(np.abs(roots))]
    return X, Y

print("Generating data...")
X_np, Y_np = generate(5000)

# ---------- Normalise ----------
X_mean = X_np.mean(0);  X_std = X_np.std(0) + 1e-8
Y_mean = float(Y_np.mean()); Y_std = float(Y_np.std()) + 1e-8

X = torch.from_numpy((X_np - X_mean) / X_std)
Y = torch.from_numpy((Y_np - Y_mean) / Y_std)

# ---------- Model ----------
model = nn.Sequential(
    nn.Linear(5, 128), nn.Tanh(),
    nn.Linear(128, 128), nn.Tanh(),
    nn.Linear(128, 1)
)

opt     = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# ---------- Training: full-batch, fast, early stop ----------
print("Training...")
best_loss  = float('inf')
best_state = None
patience   = 0

for epoch in range(3000):
    pred = model(X)
    loss = loss_fn(pred, Y)
    opt.zero_grad()
    loss.backward()
    opt.step()

    l = loss.item()
    if l < best_loss - 1e-5:
        best_loss  = l
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        patience   = 0
    else:
        patience  += 1

    if epoch % 200 == 0:
        print(f"  Epoch {epoch:4d}  loss={l:.5f}")

    # decay lr at plateau
    if patience == 300:
        for g in opt.param_groups: g['lr'] *= 0.5
        patience = 0
        print(f"  lr -> {opt.param_groups[0]['lr']:.2e}")

    if opt.param_groups[0]['lr'] < 1e-5:
        print(f"  Early stop at epoch {epoch}")
        break

model.load_state_dict(best_state)
print(f"Best loss: {best_loss:.5f}")

# ---------- Helper ----------
def ml_guess(c_np):
    c_t = torch.from_numpy((c_np.astype(np.float32) - X_mean) / X_std)
    with torch.no_grad():
        return model(c_t).item() * Y_std + Y_mean

# ---------- Qualitative test ----------
print("\n--- Qualitative test (5 samples) ---\n")
for _ in range(5):
    roots = np.random.uniform(-5, 5, 4)
    c     = np.poly(roots)
    x_p, s_p = newton(c, 0.0)
    x_ml     = ml_guess(c)
    x_h, s_h = newton(c, x_ml)
    tgt      = roots[np.argmin(np.abs(roots))]
    print(f"True roots    : {np.sort(roots)}")
    print(f"Plain  Newton : x={x_p:.6f}  f(x)={poly(x_p,c):.2e}  steps={s_p}")
    print(f"ML guess      : {x_ml:.6f}  (error={abs(x_ml-tgt):.4f})")
    print(f"Hybrid Newton : x={x_h:.6f}  f(x)={poly(x_h,c):.2e}  steps={s_h}")
    print()

# ---------- Benchmark ----------
TEST_SAMPLES = 2000
MAX_ITERS    = 20
plain_time = hybrid_time = 0.0
plain_steps = hybrid_steps = 0
plain_fails = hybrid_fails = 0

print(f"--- Benchmark ({TEST_SAMPLES} samples) ---\n")
for _ in range(TEST_SAMPLES):
    roots = np.random.uniform(-5, 5, 4)
    c     = np.poly(roots)

    t0 = time.perf_counter()
    x_p, s_p = newton(c, 0.0, iters=MAX_ITERS)
    plain_time  += time.perf_counter() - t0
    plain_steps += s_p
    if abs(poly(x_p, c)) > 1e-6: plain_fails += 1

    t0 = time.perf_counter()
    x_h, s_h = newton(c, ml_guess(c), iters=MAX_ITERS)
    hybrid_time  += time.perf_counter() - t0
    hybrid_steps += s_h
    if abs(poly(x_h, c)) > 1e-6: hybrid_fails += 1

print(f"{'':30s} {'Plain Newton':>15}  {'ML + Newton':>12}")
print(f"{'Avg time per sample (ms)':30s} {plain_time/TEST_SAMPLES*1000:>15.4f}  {hybrid_time/TEST_SAMPLES*1000:>12.4f}")
print(f"{'Avg iterations':30s} {plain_steps/TEST_SAMPLES:>15.2f}  {hybrid_steps/TEST_SAMPLES:>12.2f}")
print(f"{'Convergence failures':30s} {plain_fails:>15d}  {hybrid_fails:>12d}")
print()
print(f"Iteration reduction : {plain_steps/hybrid_steps:.2f}x")
print(f"Time speedup        : {plain_time/hybrid_time:.2f}x  (includes NN inference overhead)")
