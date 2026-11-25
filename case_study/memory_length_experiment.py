"""Lecture 21 case study: how far back can each cell remember?
Plain RNN vs LSTM vs GRU, trained with JAX.  Run:  pip install jax numpy ; python memory_length_experiment.py results.json

Sequence: first symbol is a class in {0..7}; then T-1 distractor symbols from {8..15}.
Target: the first symbol, read out from the final hidden state.
"""
import json, sys, time
import jax, jax.numpy as jnp
import numpy as np

V, C, H = 16, 8, 32

def init(kind, key, gate_bias=True):
    k = jax.random.split(key, 4)
    g = {"rnn": 1, "gru": 3, "lstm": 4}[kind]
    s = 1.0 / np.sqrt(H)
    p = {
        "Wx": jax.random.uniform(k[0], (V, g * H), minval=-s, maxval=s),
        "Wh": jax.random.uniform(k[1], (H, g * H), minval=-s, maxval=s),
        "b": jnp.zeros(g * H),
        "Wo": jax.random.uniform(k[2], (H, C), minval=-s, maxval=s),
        "bo": jnp.zeros(C),
    }
    if kind == "lstm" and gate_bias:
        p["b"] = p["b"].at[H:2 * H].set(3.0)   # forget-gate bias +3: start out remembering
    if kind == "gru" and gate_bias:
        p["b"] = p["b"].at[H:2 * H].set(-3.0)  # update-gate bias -3: start out keeping h
    return p

def step_fn(kind, p):
    def rnn(h, x):
        return jnp.tanh(x @ p["Wx"] + h @ p["Wh"] + p["b"]), None
    def lstm(carry, x):
        h, c = carry
        z = x @ p["Wx"] + h @ p["Wh"] + p["b"]
        i, f, g, o = jnp.split(z, 4, axis=-1)
        c = jax.nn.sigmoid(f) * c + jax.nn.sigmoid(i) * jnp.tanh(g)
        h = jax.nn.sigmoid(o) * jnp.tanh(c)
        return (h, c), None
    def gru(h, x):
        zx = x @ p["Wx"] + p["b"]
        zr_x, zz_x, zn_x = jnp.split(zx, 3, axis=-1)
        Whr, Whz, Whn = jnp.split(p["Wh"], 3, axis=-1)
        r = jax.nn.sigmoid(zr_x + h @ Whr)
        u = jax.nn.sigmoid(zz_x + h @ Whz)
        n = jnp.tanh(zn_x + (r * h) @ Whn)
        return (1 - u) * h + u * n, None
    return {"rnn": rnn, "lstm": lstm, "gru": gru}[kind]

def forward(kind, p, X):  # X: (B, T, V)
    B = X.shape[0]
    h0 = jnp.zeros((B, H))
    carry = (h0, h0) if kind == "lstm" else h0
    carry, _ = jax.lax.scan(step_fn(kind, p), carry, jnp.swapaxes(X, 0, 1))
    h = carry[0] if kind == "lstm" else carry
    return h @ p["Wo"] + p["bo"]

def batch(rng, B, T):
    first = rng.integers(0, C, B)
    rest = rng.integers(C, V, (B, T - 1))
    seq = np.concatenate([first[:, None], rest], 1)
    return jax.nn.one_hot(seq, V), first

def run(kind, T, seed, steps=5000, B=64, lr=3e-3, gate_bias=True):
    key = jax.random.PRNGKey(seed)
    p = init(kind, key, gate_bias)
    rng = np.random.default_rng(seed)
    m = jax.tree_util.tree_map(jnp.zeros_like, p); v = jax.tree_util.tree_map(jnp.zeros_like, p)

    def loss(p, X, y):
        lg = forward(kind, p, X)
        return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(lg), y[:, None], 1))

    @jax.jit
    def update(p, m, v, X, y, t):
        l, g = jax.value_and_grad(loss)(p, X, y)
        gn = jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g)))
        g = jax.tree_util.tree_map(lambda x: x * jnp.minimum(1.0, 5.0 / (gn + 1e-8)), g)  # clip
        m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
        v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b * b, v, g)
        p = jax.tree_util.tree_map(lambda w, a, b: w - lr * (a / (1 - 0.9 ** t)) / (jnp.sqrt(b / (1 - 0.999 ** t)) + 1e-8), p, m, v)
        return p, m, v, l

    for t in range(1, steps + 1):
        X, y = batch(rng, B, T)
        p, m, v, l = update(p, m, v, X, jnp.array(y), t)
    X, y = batch(np.random.default_rng(999), 2000, T)
    acc = float(jnp.mean(jnp.argmax(forward(kind, p, X), -1) == y))
    return acc

if __name__ == "__main__":
    configs = [("rnn", True, [10, 20, 40, 80, 160]),
               ("lstm", True, [20, 40, 80, 160]),
               ("gru", True, [20, 40, 80, 160]),
               ("gru", False, [10, 20, 40, 80, 160])]   # the first attempt: update-gate bias left at 0
    out = {}
    for kind, gb, Ts in configs:
        for T in Ts:
            accs = [run(kind, T, s, gate_bias=gb) for s in range(2)]
            name = f"{kind}{'' if gb or kind == 'rnn' else '_bias0'}_T{T}"
            out[name] = accs
            print(name, [round(100 * a) for a in accs], flush=True)
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "memory_results.json", "w"), indent=1)
