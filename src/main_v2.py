#!/bin/python

import torch
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def poly(x, c):
    return (c[0]*x**4+c[1]*x**3+c[2]*x**2+c[3]*x+c[4])

def dpoly(x, c):
    return (4*c[0]*x**3 + 3*c[1]*x**2 + 2*c[2]*x + c[3])

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

N=5000
X = np.zeros((N, 5), dtype=np.float32)
Y = np.zeros((N, 1), dtype=np.float32)

def generate(N):
    for i in range(N):
        roots = np.random.uniform(-5, 5, 4)
        c     = np.poly(roots)
        X[i]  = c
        Y[i]  = roots[np.argmin(roots)]
    return X, Y

print("Generating data...")
X_np, Y_np = generate(5000)


X_mean = np.mean(X_np,0)
X_std = np.std(X_np,0) + 1e-8     #adding a small constant to prevent divide by 0
Y_mean = float(np.mean(Y_np))
Y_std = float(np.std(Y_np)) + 1e-8

X = torch.from_numpy((X_np - X_mean) / X_std)
Y = torch.from_numpy((Y_np - Y_mean) / Y_std)


model = nn.Sequential(
    nn.Linear(5, 128), nn.Tanh(),
    nn.Linear(128, 128), nn.Tanh(),
    nn.Linear(128, 1)
)

opt     = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()


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

    if patience == 300:
        for g in opt.param_groups: g['lr'] *= 0.5
        patience = 0
        print(f"  lr -> {opt.param_groups[0]['lr']:.2e}")

    if opt.param_groups[0]['lr'] < 1e-5:
        print(f"  Early stop at epoch {epoch}")
        break

model.load_state_dict(best_state)
print(f"Best loss: {best_loss:.5f}")


print("\n--- Qualitative test (5 samples) ---\n")
for _ in range(5):
    roots = np.random.uniform(-5, 5, 4)
    c     = np.poly(roots)
    x_p, s_p = newton(c, 0.0)
    c_t  = torch.from_numpy((c.astype(np.float32) - X_mean) / X_std)
    with torch.no_grad():
        x_ml = model(c_t).item() * Y_std + Y_mean
    x_h, s_h = newton(c, x_ml)
    tgt  = roots[np.argmin((roots))]
    print(f"True roots    : {np.sort(roots)}")
    print(f"Plain  Newton : x={x_p:.6f}  f(x)={poly(x_p,c):.2e}  steps={s_p}")
    print(f"ML guess      : {x_ml:.6f}  (error={abs(x_ml-tgt):.4f})")
    print(f"Hybrid Newton : x={x_h:.6f}  f(x)={poly(x_h,c):.2e}  steps={s_h}")
    print()


TEST_SAMPLES = 2000
MAX_ITERS    = 20

# generate all test cases upfront
all_roots = [np.random.uniform(-5, 5, 4) for _ in range(TEST_SAMPLES)]
all_c     = [np.poly(r) for r in all_roots]
all_c_np  = np.array(all_c, dtype=np.float32)

# ---- plain Newton (sequential, starting from 0) ----
plain_steps = 0
plain_fails = 0
t0 = time.perf_counter()
plain_results = []
for c in all_c:
    x, s = newton(c, 0.0, iters=MAX_ITERS)
    plain_results.append(x)
    plain_steps += s
    if abs(poly(x, c)) > 1e-6: plain_fails += 1
plain_time = time.perf_counter() - t0

# ---- ML warm start: ONE batched forward pass, then sequential Newton ----
hybrid_steps = 0
hybrid_fails = 0
t0 = time.perf_counter()

# single forward pass for all 2000 samples at once
C_norm = torch.from_numpy((all_c_np - X_mean) / X_std)
with torch.no_grad():
    x0_all = model(C_norm).numpy().flatten() * Y_std + Y_mean

# Newton polish for each sample
for c, x0 in zip(all_c, x0_all):
    x, s = newton(c, float(x0), iters=MAX_ITERS)
    hybrid_steps += s
    if abs(poly(x, c)) > 1e-6: hybrid_fails += 1
hybrid_time = time.perf_counter() - t0

# ---- Results ----
print(f"--- Benchmark ({TEST_SAMPLES} samples) ---\n")
print(f"{'':30s} {'Plain Newton':>15}  {'ML + Newton':>12}")
print(f"{'Total time (ms)':30s} {plain_time*1000:>15.2f}  {hybrid_time*1000:>12.2f}")
print(f"{'Avg time per sample (ms)':30s} {plain_time/TEST_SAMPLES*1000:>15.4f}  {hybrid_time/TEST_SAMPLES*1000:>12.4f}")
print(f"{'Avg iterations':30s} {plain_steps/TEST_SAMPLES:>15.2f}  {hybrid_steps/TEST_SAMPLES:>12.2f}")
print(f"{'Convergence failures':30s} {plain_fails:>15d}  {hybrid_fails:>12d}")
print()
print(f"Iteration reduction : {plain_steps/hybrid_steps:.2f}x")
print(f"Time speedup        : {plain_time/hybrid_time:.2f}x")
print(f"  (NN did 1 forward pass for all {TEST_SAMPLES} samples combined)")
