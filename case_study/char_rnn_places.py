"""
Case study (Lecture 20): a character-level RNN that invents Kerala place names.

Pure NumPy, about 100 lines, so every step of backpropagation through time is visible.
Run:  python char_rnn_places.py kerala_places.txt
Needs only numpy. Takes under a minute on a laptop.
"""
import sys
import numpy as np

# ---------------- data ----------------
names = [l.strip().lower() for l in open(sys.argv[1] if len(sys.argv) > 1 else "kerala_places.txt") if l.strip()]
chars = sorted(set("".join(names)))
itos = ["."] + chars                      # "." marks both the start and the end of a name
stoi = {c: i for i, c in enumerate(itos)}
V, H = len(itos), 100                     # vocabulary size, hidden size
print(f"{len(names)} names, vocabulary of {V} symbols")

rng = np.random.default_rng(42)
Wxh = rng.normal(0, 0.01, (H, V))         # input  -> hidden
Whh = rng.normal(0, 0.01, (H, H))         # hidden -> hidden   (the recurrence)
Why = rng.normal(0, 0.01, (V, H))         # hidden -> output
bh, by = np.zeros((H, 1)), np.zeros((V, 1))
params = [Wxh, Whh, Why, bh, by]
mem = [np.zeros_like(p) for p in params]  # Adagrad memory


def loss_and_grads(inputs, targets, hprev):
    """Forward pass over one name, then backpropagation through time."""
    xs, hs, ps = {}, {-1: hprev}, {}
    loss = 0.0
    for t in range(len(inputs)):                                  # ---- forward
        xs[t] = np.zeros((V, 1)); xs[t][inputs[t]] = 1
        hs[t] = np.tanh(Wxh @ xs[t] + Whh @ hs[t - 1] + bh)       # h_t = tanh(W_xh x_t + W_hh h_{t-1} + b)
        y = Why @ hs[t] + by
        ps[t] = np.exp(y - y.max()); ps[t] /= ps[t].sum()        # softmax over next character
        loss -= np.log(ps[t][targets[t], 0])                      # cross-entropy
    grads = [np.zeros_like(p) for p in params]
    dWxh, dWhh, dWhy, dbh, dby = grads
    dh_next = np.zeros((H, 1))
    for t in reversed(range(len(inputs))):                        # ---- backward through time
        dy = ps[t].copy(); dy[targets[t]] -= 1
        dWhy += dy @ hs[t].T; dby += dy
        dh = Why.T @ dy + dh_next                                 # gradient arriving from the future
        draw = (1 - hs[t] ** 2) * dh                              # back through tanh
        dWxh += draw @ xs[t].T; dWhh += draw @ hs[t - 1].T; dbh += draw
        dh_next = Whh.T @ draw                                    # pass it one step further back
    for g in grads:
        np.clip(g, -5, 5, out=g)                                  # gradient clipping
    return loss / len(inputs), grads


def sample(n, temperature=0.8, seed=0):
    r = np.random.default_rng(seed); out = []
    for _ in range(n):
        h = np.zeros((H, 1)); ix = 0; s = ""
        for _ in range(20):
            x = np.zeros((V, 1)); x[ix] = 1
            h = np.tanh(Wxh @ x + Whh @ h + bh)
            y = (Why @ h + by) / temperature
            p = np.exp(y - y.max()); p /= p.sum()
            ix = r.choice(V, p=p.ravel())
            if ix == 0: break
            s += itos[ix]
        out.append(s)
    return out


checkpoints = {0, 500, 2000, 8000, 30000}
smooth = None
for it in range(30001):
    if it in checkpoints:
        print(f"\n--- after {it} names (loss {smooth if smooth else float('nan'):.2f}) ---")
        print(", ".join(sample(8, seed=7)))
    if it == 30000: break
    name = names[rng.integers(len(names))]
    ids = [0] + [stoi[c] for c in name] + [0]
    loss, grads = loss_and_grads(ids[:-1], ids[1:], np.zeros((H, 1)))
    smooth = loss if smooth is None else 0.99 * smooth + 0.01 * loss
    for p, g, m in zip(params, grads, mem):                       # Adagrad update
        m += g * g
        p -= 0.1 * g / np.sqrt(m + 1e-8)

new = [s for s in sample(300, temperature=1.0, seed=11) if s not in names and len(s) > 4]
seen = sum(s in names for s in sample(300, seed=5)) / 300
print(f"\nShare of samples copied from the training list: {seen:.0%}")
print("Some names it made up:", ", ".join(sorted(set(new))[:30]))
