# ============================================================
# jx.py  —  JX: jQuery for JAX  (tight build)
# Author : J. Roberto Jiménez · tijuanapaint@gmail.com
# ============================================================
"""
Design principles
─────────────────
• The graph IS the function — node_fn is a composed closure,
  never a list that gets reduced at runtime.
• Every method returns a new JX — fully immutable chain.
• JX objects are first-class JAX callables (__call__ = node_fn).
• Execution is explicit: .value() | .jit() | .grad() | .scan()
"""

import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from jax import lax
import optax
from typing import Any, Callable, Optional, Tuple, Union
import time


# ─────────────────────────────────────────────────────────────
# Core
# ─────────────────────────────────────────────────────────────

class JX:
    """
    A single node in a lazy computation graph.

    data    — leaf input (JAX array or None for reusable pipelines)
    node_fn — accumulated composed function  f = fN ∘ … ∘ f1
    name    — human-readable trace of the chain
    """

    __slots__ = ("data", "node_fn", "name", "_is_vmapped", "_in_axes", "_out_axes", "_is_pmapped")

    def __init__(
        self,
        data: Any,
        node_fn: Callable = None,
        name: str = "root",
    ):
        self.data    = data
        self.node_fn = node_fn if node_fn is not None else (lambda x: x)
        self.name    = name
        self._is_vmapped = False
        self._in_axes = 0
        self._out_axes = 0
        self._is_pmapped = False

    # ── Internal ─────────────────────────────────────────────

    def _chain(self, fn: Callable, label: str) -> "JX":
        """Compose fn onto the existing pipeline — O(1), no list."""
        prev = self.node_fn
        if getattr(self, "_is_vmapped", False):
            # If we are in 'vmap mode', we wrap the next function
            new_fn = lambda x: vmap(fn, in_axes=self._in_axes, out_axes=self._out_axes)(prev(x))
        elif getattr(self, "_is_pmapped", False):
            # If we are in 'pmap mode', we wrap the next function
            new_fn = lambda x: jax.pmap(fn)(prev(x))
        else:
            new_fn = lambda x: fn(prev(x))

        
        new_jx = JX(self.data, new_fn, f"{self.name}→{label}")
        if getattr(self, "_is_vmapped", False):
            new_jx._set_vmap(self._in_axes, self._out_axes)
        if getattr(self, "_is_pmapped", False):
            new_jx._set_pmap()
        return new_jx

    def _label(self, fn: Callable, fallback: str) -> str:
        return getattr(fn, "__name__", fallback)

    def _set_vmap(self, in_axes, out_axes):
        self._is_vmapped = True
        self._in_axes = in_axes
        self._out_axes = out_axes
        return self

    def _set_pmap(self):
        self._is_pmapped = True
        return self

    # ── Make JX a first-class JAX function ───────────────────

    def __call__(self, x: Any) -> Any:
        """Apply the pipeline to an arbitrary input (no .data needed)."""
        return self.node_fn(x)

    # ─────────────────────────────────────────────────────────
    # Fluent API
    # ─────────────────────────────────────────────────────────

    def pipe(self, fn: Callable) -> "JX":
        """Inject any function into the chain."""
        return self._chain(fn, self._label(fn, "pipe"))

    # ── Execution strategies ──────────────────────────────────

    def value(self) -> Any:
        """Materialize the graph — eager execution."""
        if self.data is None:
            raise ValueError("No data attached. Use jx(data) or call the pipeline directly.")
        return self.node_fn(self.data)

    def value_of(self) -> Any:
        """Alias for .value() to match README/Notebook."""
        return self.value()

    def fuse(self) -> "JX":
        """
        JIT-compile the accumulated pipeline into a single XLA op.
        Returns a new JX with the compiled function baked in.
        """
        compiled = jit(self.node_fn)
        return JX(self.data, compiled, f"jit({self.name})")

    def map(self, in_axes: int = 0, out_axes: int = 0) -> "JX":
        """
        Vectorize the pipeline over a batch dimension.
        Subsequent operations in the chain will be vmapped.
        """
        return JX(
            self.data,
            self.node_fn,
            f"vmap({self.name})",
        )._set_vmap(in_axes, out_axes)

    def vmap(self, in_axes: int = 0, out_axes: int = 0) -> "JX":
        """Alias for .map() to match README/Notebook."""
        return self.map(in_axes=in_axes, out_axes=out_axes)

    def pmap(self) -> "JX":
        """Split work across multiple devices/GPUs."""
        return JX(
            self.data,
            self.node_fn,
            f"pmap({self.name})",
        )._set_pmap()

    def scan(
        self,
        fn: Callable[[Any, Any], Tuple[Any, Any]],
        init: Any,
        *,
        length: Optional[int] = None,
        reverse: bool = False,
    ) -> "JX":
        """
        Apply jax.lax.scan over the pipeline's output (treated as xs).

        fn    : (carry, x) → (carry, y)   — pure JAX function
        init  : initial carry
        length: optional explicit length

        Returns a JX whose value is (final_carry, stacked_ys).

        Example — running sum:
            jx(xs).scan(lambda c, x: (c + x, c + x), 0.0).value()

        Example — Euler ODE step:
            jx(ts).scan(euler_step, y0).value()
        """
        pipeline = self.node_fn

        def _scan_node(x):
            xs = pipeline(x)
            return lax.scan(fn, init, xs, length=length, reverse=reverse)

        return JX(self.data, _scan_node, f"scan({self.name})")

    # ── Autograd ─────────────────────────────────────────────

    def grad(
        self,
        argnums: Union[int, Tuple] = 0,
        has_aux: bool = False,
    ) -> "JX":
        """
        Differentiate the pipeline.
        Returns a JX that computes ∇(pipeline) at .data.

        Note: grad replaces the pipeline — it differentiates the
        *whole* chain up to this point.
        """
        grad_fn = grad(self.node_fn, argnums=argnums, has_aux=has_aux)
        return JX(self.data, grad_fn, f"∇({self.name})")

    def value_and_grad(self, argnums: int = 0) -> "JX":
        """
        Returns a JX whose value is (output, gradient).
        Single forward+backward pass — no double computation.
        """
        vg = jax.value_and_grad(self.node_fn, argnums=argnums)
        return JX(self.data, vg, f"val+∇({self.name})")

    def hessian(self) -> "JX":
        fn = self.node_fn
        return JX(self.data, jax.hessian(fn), f"H({self.name})")

    def jacrev(self) -> "JX":
        fn = self.node_fn
        return JX(self.data, jax.jacrev(fn), f"Jrev({self.name})")

    def jacfwd(self) -> "JX":
        fn = self.node_fn
        return JX(self.data, jax.jacfwd(fn), f"Jfwd({self.name})")

    # ── Shape ─────────────────────────────────────────────────

    def reshape(self, *shape)  -> "JX": return self._chain(lambda x: x.reshape(*shape),    "reshape")
    def transpose(self, *axes) -> "JX": return self._chain(lambda x: x.transpose(*axes),   "T")
    def squeeze(self, axis=None)->"JX": return self._chain(lambda x: x.squeeze(axis=axis), "squeeze")
    def expand(self, axis: int) -> "JX": return self._chain(lambda x: jnp.expand_dims(x, axis), "expand")
    def __getitem__(self, idx)  -> "JX": return self._chain(lambda x: x[idx],              "[]")

    # ── Activations ───────────────────────────────────────────

    def relu(self)      -> "JX": return self._chain(jax.nn.relu,        "relu")
    def sigmoid(self)   -> "JX": return self._chain(jax.nn.sigmoid,     "sigmoid")
    def log_sigmoid(self)->"JX": return self._chain(jax.nn.log_sigmoid, "log_σ")
    def tanh(self)      -> "JX": return self._chain(jnp.tanh,           "tanh")
    def gelu(self)      -> "JX": return self._chain(jax.nn.gelu,        "gelu")
    def softmax(self, axis=-1) -> "JX":
        return self._chain(lambda x: jax.nn.softmax(x, axis=axis), "softmax")
    def log_softmax(self, axis=-1) -> "JX":
        return self._chain(lambda x: jax.nn.log_softmax(x, axis=axis), "log_sm")

    # ── Elementwise math ──────────────────────────────────────

    def abs(self)  -> "JX": return self._chain(jnp.abs,  "abs")
    def sqrt(self) -> "JX": return self._chain(jnp.sqrt, "sqrt")
    def exp(self)  -> "JX": return self._chain(jnp.exp,  "exp")
    def log(self)  -> "JX": return self._chain(jnp.log,  "log")
    def sin(self)  -> "JX": return self._chain(jnp.sin,  "sin")
    def cos(self)  -> "JX": return self._chain(jnp.cos,  "cos")

    def normalize(self, eps: float = 1e-8) -> "JX":
        def _n(x): return (x - x.mean()) / jnp.sqrt(x.var() + eps)
        return self._chain(_n, "norm")

    # ── Reductions ────────────────────────────────────────────

    def sum(self, axis=None, keepdims=False) -> "JX":
        return self._chain(lambda x: jnp.sum(x, axis=axis, keepdims=keepdims), "sum")
    def mean(self, axis=None, keepdims=False) -> "JX":
        return self._chain(lambda x: jnp.mean(x, axis=axis, keepdims=keepdims), "mean")
    def var(self, axis=None, keepdims=False) -> "JX":
        return self._chain(lambda x: jnp.var(x, axis=axis, keepdims=keepdims), "var")
    def std(self, axis=None, keepdims=False) -> "JX":
        return self._chain(lambda x: jnp.std(x, axis=axis, keepdims=keepdims), "std")
    def max(self, axis=None, keepdims=False) -> "JX":
        return self._chain(lambda x: jnp.max(x, axis=axis, keepdims=keepdims), "max")
    def min(self, axis=None, keepdims=False) -> "JX":
        return self._chain(lambda x: jnp.min(x, axis=axis, keepdims=keepdims), "min")

    # ── Binary operators ──────────────────────────────────────
    # _resolve: if other is a JX, evaluate its branch on the same x
    # (shared-input branching for residuals, attention, etc.)

    @staticmethod
    def _resolve(other: Any, x: Any) -> Any:
        if isinstance(other, JX):
            return other.node_fn(x)   # branch shares same x
        return other

    def __add__(self, o)  -> "JX": return self._chain(lambda x: x + JX._resolve(o, x), "add")
    def __radd__(self, o) -> "JX": return self._chain(lambda x: JX._resolve(o, x) + x, "radd")
    def __sub__(self, o)  -> "JX": return self._chain(lambda x: x - JX._resolve(o, x), "sub")
    def __rsub__(self, o) -> "JX": return self._chain(lambda x: JX._resolve(o, x) - x, "rsub")
    def __mul__(self, o)  -> "JX": return self._chain(lambda x: x * JX._resolve(o, x), "mul")
    def __rmul__(self, o) -> "JX": return self._chain(lambda x: JX._resolve(o, x) * x, "rmul")
    def __truediv__(self, o)->"JX":return self._chain(lambda x: x / JX._resolve(o, x), "div")
    def __pow__(self, o)  -> "JX": return self._chain(lambda x: x ** JX._resolve(o, x),"pow")
    def __matmul__(self, o)->"JX": return self._chain(lambda x: x @ JX._resolve(o, x), "@")
    def __neg__(self)     -> "JX": return self._chain(lambda x: -x,                     "neg")

    def dot(self, other)    -> "JX": return self @ other
    def matmul(self, other) -> "JX": return self @ other

    def einsum(self, subscripts: str, *operands) -> "JX":
        def _e(x):
            ops = [o.node_fn(x) if isinstance(o, JX) else o for o in operands]
            return jnp.einsum(subscripts, x, *ops)
        return self._chain(_e, f"ein:{subscripts}")

    # ── Debug ─────────────────────────────────────────────────

    def tap(self, msg: str = "") -> "JX":
        """Passthrough that prints shape/dtype/mean — for mid-chain debugging."""
        def _t(x):
            print(f"[jx·tap] {msg:20s}  shape={x.shape}  dtype={x.dtype}  μ={float(jnp.mean(x)):.4f}")
            return x
        return self._chain(_t, f"tap")

    def inspect(self) -> dict:
        return {
            "name":     self.name,
            "has_data": self.data is not None,
            "callable": True,
        }

    def __repr__(self) -> str:
        tag = "✓" if self.data is not None else "∅"
        return f"JX{tag}[ {self.name} ]"


# ─────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────

def jx(data=None, name: str = "root") -> JX:
    """
    Entry point — the $ of JAX.

    jx(array)   — rooted pipeline, .value() works immediately
    jx()        — reusable pipeline, call it like a function
    """
    return JX(data, name=name)

jq = jx   # jQuery alias  ($ is not valid Python syntax at module level — use jq or import as $)


# ─────────────────────────────────────────────────────────────
# Layers
# ─────────────────────────────────────────────────────────────

class Linear:
    """
    Dense layer.  Parameters are plain JAX arrays — no pytree magic,
    intentionally simple so they compose cleanly with JX pipelines.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        key=None,
    ):
        key   = key if key is not None else jax.random.PRNGKey(0)
        scale = jnp.sqrt(2.0 / (in_features + out_features))
        self.W = jax.random.normal(key, (in_features, out_features)) * scale
        self.b = jnp.zeros(out_features) if use_bias else None

    def __call__(self, x: JX) -> JX:
        """Returns a new JX with matmul + bias appended to the chain."""
        result = x @ self.W
        return result + self.b if self.b is not None else result

    def parameters(self):
        p = [jx(self.W, name="W")]
        if self.b is not None:
            p.append(jx(self.b, name="b"))
        return p


class Sequential:
    """Stack of callable layers, each receiving and returning a JX."""

    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x: JX) -> JX:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for l in self.layers
                if hasattr(l, "parameters")
                for p in l.parameters()]


# ─────────────────────────────────────────────────────────────
# Optimizer
# ─────────────────────────────────────────────────────────────

class Optimizer:
    """
    Thin stateful wrapper around optax.
    Operates on a flat JAX array of parameters for simplicity.
    """

    def __init__(self, params: JX, tx: optax.GradientTransformation):
        if params.data is None:
            raise ValueError("Optimizer needs a JX with data attached.")
        self.params    = params
        self.tx        = tx
        self.opt_state = tx.init(params.data)

    def step(self, loss_fn: Callable) -> Tuple[float, "Optimizer"]:
        """Single gradient step — one value_and_grad call, no redundancy."""
        val, grads = jax.value_and_grad(loss_fn)(self.params.data)
        updates, new_state = self.tx.update(grads, self.opt_state, self.params.data)
        new_p = optax.apply_updates(self.params.data, updates)

        new_opt            = Optimizer.__new__(Optimizer)
        new_opt.params     = JX(new_p, name="params")
        new_opt.tx         = self.tx
        new_opt.opt_state  = new_state
        return float(val), new_opt

    def train(
        self,
        loss_fn: Callable,
        steps: int = 100,
        callback: Callable = None,
    ):
        losses, cur = [], self
        for i in range(steps):
            loss, cur = cur.step(loss_fn)
            losses.append(loss)
            if callback:
                callback(i, loss, cur.params)
        # mutate self so the caller doesn't need to reassign
        self.params    = cur.params
        self.opt_state = cur.opt_state
        return losses


def sgd(params: JX, lr: float = 0.01)   -> Optimizer: return Optimizer(params, optax.sgd(lr))
def adam(params: JX, lr: float = 0.001) -> Optimizer: return Optimizer(params, optax.adam(lr))
def adamw(params: JX, lr: float = 0.001)-> Optimizer: return Optimizer(params, optax.adamw(lr))


# ─────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────

def mse(pred: JX, target: JX)  -> JX: return ((pred - target) ** 2).mean()
def mae(pred: JX, target: JX)  -> JX: return (pred - target).abs().mean()

def cross_entropy(logits: JX, labels: JX) -> JX:
    return -(labels * logits.log_softmax()).sum()

def binary_cross_entropy(logits: JX, labels: JX) -> JX:
    return -(
        labels * logits.log_sigmoid()
        + (1 - labels) * (-logits).log_sigmoid()
    ).mean()


# ─────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────

def _time(fn, *args, repeats=3, label=""):
    """Minimal timer — warms up once, then averages."""
    fn(*args)  # warm-up / JIT compile
    jax.block_until_ready(fn(*args))
    t0 = time.perf_counter()
    for _ in range(repeats):
        jax.block_until_ready(fn(*args))
    ms = (time.perf_counter() - t0) / repeats * 1000
    print(f"  {label:35s}  {ms:7.3f} ms")
    return ms


def demo():
    sep = "─" * 60

    # ── 0. Basics ────────────────────────────────────────────
    print(sep)
    print("JX  tight build")
    print(sep)

    x = jnp.array([1.0, -2.0, 3.0, -4.0, 5.0])

    pipeline = jq(x).normalize().relu().softmax()
    print("\n[pipeline]", pipeline)
    print("value()  :", pipeline.value())

    # JX as a callable (no .data needed)
    reusable =  jq(None).normalize().relu().softmax()
    print("\n[reusable pipeline applied to x]")
    print("         :", reusable(x))

    # ── 1. Branching (residual-style) ─────────────────────────
    print(f"\n{sep}")
    print("Branching  —  a + b  shares the same x")
    a =  jq(x).normalize()
    b =  jq(x).relu()
    c = a + b           # both branches evaluated from the same x
    print("a + b    :", c.value())

    # ── 2. .fuse() — JIT ────────────────────────────────────
    print(f"\n{sep}")
    print("Fusion  —  .fuse() compiles the whole chain")
    deep =  jq(x).normalize().relu().softmax().log().sum()
    fused = deep.fuse()
    print("eager    :", deep.value())
    print("fused    :", fused.value())

    # ── 3. .map() — vmap ─────────────────────────────────────
    print(f"\n{sep}")
    print("Map  —  .map() vmaps over batch dim 0")
    batch = jnp.stack([x, x * 2, x * 3])
    result =  jq(batch).map().normalize().relu().sum().value()
    print("vmap(normalize→relu→sum) per sample:", result)

    # ── 4. .scan() ───────────────────────────────────────────
    print(f"\n{sep}")
    print("Scan  —  three demos")

    # 4a. Cumulative sum
    xs = jnp.ones(1_000)
    carry, ys =  jq(xs).scan(lambda c, x: (c + x, c + x), 0.0).value()
    print(f"\n  [cumsum]  final carry={carry:.0f}  (expected 1000)")

    # 4b. Euler ODE   dy/dt = -y,  y(0)=1  →  y(t)=e^-t
    dt = 0.001
    ts = jnp.arange(1_000) * dt
    def euler_step(y, t):
        return y + dt * (-y), y          # carry=new y, output=old y
    _, ys_ode =  jq(ts).scan(euler_step, 1.0).value()
    print(f"  [ODE]     y(1.0) ≈ {ys_ode[-1]:.5f}  (expected {jnp.exp(-1.0):.5f})")

    # 4c. Vanilla RNN
    T, D_in, D_h = 20, 4, 8
    key = jax.random.PRNGKey(0)
    Wx = jax.random.normal(key, (D_in, D_h)) * 0.1
    Wh = jax.random.normal(key, (D_h,  D_h)) * 0.1
    seq = jax.random.normal(key, (T, D_in))

    def rnn_step(h, x):
        h_new = jnp.tanh(x @ Wx + h @ Wh)
        return h_new, h_new

    h0 = jnp.zeros(D_h)
    h_final, h_seq =  jq(seq).scan(rnn_step, h0).value()
    print(f"  [RNN]     h_seq shape={h_seq.shape}  |h_final|={jnp.linalg.norm(h_final):.4f}")

    # ── 5. scan timing ───────────────────────────────────────
    print(f"\n{sep}")
    print("Timing  —  Python loop vs scan vs scan+fuse  (N=10_000)")

    N   = 10_000
    xs_big = jnp.ones(N)

    def python_loop(xs):
        c = 0.0
        for i in range(N):
            c = c + xs[i]
        return c

    scan_pipeline      =  jq(xs_big).scan(lambda c, x: (c + x, c + x), 0.0)
    scan_fused_pipeline=  jq(xs_big).scan(lambda c, x: (c + x, c + x), 0.0).fuse()

    print()
    _time(python_loop,              xs_big, label="Python loop  (N=10k)")
    _time(scan_pipeline.value,              label=".scan()      (N=10k)")
    _time(scan_fused_pipeline.value,        label=".scan().fuse()(N=10k)")

    # ── 6. Autograd ──────────────────────────────────────────
    print(f"\n{sep}")
    print("Autograd")

    g =  jq(x).normalize().relu().sum().grad().value()
    print(f"\n  grad  of normalize→relu→sum : {g}")

    val, g2 =  jq(x).normalize().relu().sum().value_and_grad().value()
    print(f"  value                       : {val:.4f}")
    print(f"  grad  (value_and_grad)      : {g2}")

    # ── 7. Optimizer ─────────────────────────────────────────
    print(f"\n{sep}")
    print("Optimizer  —  converge params → [3, 3, 3]")

    target = jnp.array([3.0, 3.0, 3.0])
    def loss_fn(p): return jnp.mean((p - target) ** 2)

    opt    = adam( jq(jnp.zeros(3)), lr=0.05)
    losses = opt.train(loss_fn, steps=200)
    print(f"  final params : {opt.params.data}")
    print(f"  final loss   : {losses[-1]:.6f}")

    # ── 8. Linear layer ──────────────────────────────────────
    print(f"\n{sep}")
    print("Linear layer  →  Sequential MLP")

    k1, k2 = jax.random.split(jax.random.PRNGKey(1))
    mlp = Sequential(
        Linear(5, 16, key=k1),
        lambda x: x.relu(),
        Linear(16, 1,  key=k2),
    )
    out = mlp( jq(x)).value()
    print(f"  MLP(x) = {out}")

    print(f"\n{sep}")
    print("Done.")


if __name__ == "__main__":
    demo()
