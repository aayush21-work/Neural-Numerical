#!/bin/python

import torch
import time
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ---------- polynomial ----------
def poly(x, c):
    s = 0
    for _ in range(200):
        s += (((c[0]*x + c[1])*x + c[2])*x + c[3])*x + c[4]
    return s / 200

def dpoly(x, c):
    return (4*c[0]*x**3 + 3*c[1]*x**2 + 2*c[2]*x + c[3])

# ---------- Newton ----------
def newton(c, x0, iters=20):
    x = x0
    for _ in range(iters):
        fx = poly(x, c)
        dfx = dpoly(x, c)
        if abs(dfx) < 1e-10:
            break
        x = x - fx/dfx
    return x

# ---------- Generate quartic dataset ----------
def generate(N=6000):
    X = np.zeros((N, 5))
    Y = np.zeros((N, 1))

    for i in range(N):
        roots = np.random.uniform(-5, 5, 4)
        c = np.poly(roots)   # [a4,a3,a2,a1,a0]

        target = roots[np.argmin(np.abs(roots))]

        X[i] = c
        Y[i] = target

    return torch.from_numpy(X).float(), \
           torch.from_numpy(Y).float()
print("Generating data...")
X, Y = generate(8000)

# ---------- Neural Net ----------
model = nn.Sequential(
    nn.Linear(5, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1)
)

opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

print("Training...")
for epoch in range(2000):
    pred = model(X)
    loss = loss_fn(pred, Y)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % 200 == 0:
        print("Epoch", epoch, "Loss", loss.item())

# ---------- Test ----------
print("\nTesting hybrid ML + Newton\n")

for _ in range(5):
    roots = np.random.uniform(-5,5,4)
    c = np.poly(roots)

    # Plain Newton from 0
    x_plain = newton(c, 0.0)

    # ML guess
    with torch.no_grad():
        x_ml = model(torch.tensor(c, dtype=torch.float32)).item()

    x_hybrid = newton(c, x_ml)

    print("True roots:", roots)
    print("Plain Newton:", x_plain)
    print("ML guess:", x_ml)
    print("Hybrid:", x_hybrid)
    print("f(root) =", poly(x_hybrid,c))
    print()






# --------- Benchmark settings ----------
TEST_SAMPLES = 2000
NEWTON_ITERS = 20

plain_time = 0.0
hybrid_time = 0.0

plain_steps_total = 0
hybrid_steps_total = 0

print("\nRunning benchmark...\n")

c_tensor = torch.empty(5, dtype=torch.float32)

for _ in range(TEST_SAMPLES):

    # generate random quartic
    roots = np.random.uniform(-5,5,4)
    c = np.poly(roots)

    # -------- Plain Newton --------
    t0 = time.perf_counter()

    x = 0.0
    steps = 0
    for i in range(NEWTON_ITERS):
        fx = poly(x, c)
        dfx = dpoly(x, c)
        if abs(dfx) < 1e-12:
            break
        x_new = x - fx/dfx
        steps += 1
        if abs(x_new - x) < 1e-10:
            break
        x = x_new

    plain_steps_total += steps
    plain_time += time.perf_counter() - t0


    # -------- ML + Newton --------
    t0 = time.perf_counter()

    c_tensor[:] = torch.from_numpy(c)

    with torch.no_grad():
        x = model(c_tensor).item()

    steps = 0
    for i in range(NEWTON_ITERS):
        fx = poly(x, c)
        dfx = dpoly(x, c)
        if abs(dfx) < 1e-12:
            break
        x_new = x - fx/dfx
        steps += 1
        if abs(x_new - x) < 1e-10:
            break
        x = x_new

    hybrid_steps_total += steps
    hybrid_time += time.perf_counter() - t0


# -------- Results --------
print("Samples tested:", TEST_SAMPLES)
print()

print("Plain Newton:")
print("  Avg time per root:", plain_time / TEST_SAMPLES, "sec")
print("  Avg iterations   :", plain_steps_total / TEST_SAMPLES)

print("\nML + Newton:")
print("  Avg time per root:", hybrid_time / TEST_SAMPLES, "sec")
print("  Avg iterations   :", hybrid_steps_total / TEST_SAMPLES)

print("\nSpeedup (time):", plain_time / hybrid_time, "x")
print("Iteration reduction:",
      plain_steps_total / hybrid_steps_total, "x")
