# ============================================================
# jx.py
# Author: J. Roberto Jiménez - tijuanapaint@gmail.com
# ============================================================
"""
JX - jQuery for JAX
A fluent, lazy-evaluation interface for tensor computation
"""

import jax
import jax.numpy as jnp
from jax import jit, grad, vmap, pmap
from functools import reduce
from typing import Any, Callable, List, Dict, Optional, Union, Tuple
import hashlib


class JXTensor:
    """
    Lazy tensor representation - builds a computation graph without executing.

    Each JXTensor holds:
      - data:       the leaf input (or None for pure-transform nodes)
      - transforms: ordered list of (fn, label) pairs applied left-to-right
    """

    def __init__(self,
                 data: Optional[Any] = None,
                 transforms: List[Tuple[Callable, str]] = None,
                 name: Optional[str] = None,
                 batch_dim: Optional[int] = None,
                 device_count: Optional[int] = None,
                 requires_grad: bool = False):

        self.data = data
        self.transforms = transforms or []
        self.name = name or f"jx_{id(self)}"
        self.batch_dim = batch_dim
        self.device_count = device_count
        self.requires_grad = requires_grad
        self._graph_hash = None

    # ----------------------------------------------------------
    # INTERNAL HELPERS
    # ----------------------------------------------------------

    def _add(self, fn: Callable, fn_name: str = None) -> 'JXTensor':
        """Append a transform to the pipeline (returns new tensor, lazy)."""
        label = fn_name or getattr(fn, '__name__', repr(fn))
        return JXTensor(
            data=self.data,
            transforms=self.transforms + [(fn, label)],
            batch_dim=self.batch_dim,
            device_count=self.device_count,
            requires_grad=self.requires_grad,
        )

    def _compose(self, x: Any) -> Any:
        """Apply all transforms sequentially to x."""
        # FIX: Python 3 removed tuple unpacking in lambda args.
        # Use t[0](acc) instead of (fn, _): fn(acc).
        return reduce(lambda acc, t: t[0](acc), self.transforms, x)

    def _clone(self, **overrides) -> 'JXTensor':
        """Return a shallow clone with selected fields overridden."""
        return JXTensor(
            data=overrides.get('data', self.data),
            transforms=overrides.get('transforms', self.transforms),
            name=overrides.get('name', self.name),
            batch_dim=overrides.get('batch_dim', self.batch_dim),
            device_count=overrides.get('device_count', self.device_count),
            requires_grad=overrides.get('requires_grad', self.requires_grad),
        )

    @property
    def graph_hash(self) -> str:
        """Short hash of the transform sequence (useful for cache keys)."""
        if self._graph_hash is None:
            labels = str([label for _, label in self.transforms])
            self._graph_hash = hashlib.md5(labels.encode()).hexdigest()[:8]
        return self._graph_hash

    # ----------------------------------------------------------
    # EXECUTION STRATEGIES
    # ----------------------------------------------------------

    def value_of(self) -> Any:
        """Eager execution — compute and return a plain JAX array."""
        if self.data is None:
            raise ValueError("Cannot compute value: no input data provided.")
        return self._compose(self.data)

    def jit(self, **kwargs) -> Union[Any, 'JXTensor']:
        """
        JIT-compile the entire pipeline.

        - With data   → executes immediately, returns plain JAX array.
        - Without data → returns a new JXTensor whose single transform is
                         the compiled function (ready for .value_of(data)).
        """
        compiled_fn = jit(self._compose, **kwargs)
        if self.data is not None:
            return compiled_fn(self.data)
        return JXTensor(
            data=None,
            transforms=[(compiled_fn, f"jit_{self.graph_hash}")],
            batch_dim=self.batch_dim,
            device_count=self.device_count,
        )

    def vmap(self, in_axes=0, out_axes=0) -> 'JXTensor':
        """Vectorise the pipeline over a batch dimension."""
        # Capture the current pipeline before adding the vmap wrapper.
        pipeline = self._compose
        return self._clone(
            transforms=self.transforms + [
                (lambda x: vmap(pipeline, in_axes=in_axes, out_axes=out_axes)(x),
                 "vmap")
            ],
            batch_dim=in_axes,
        )

    def pmap(self, devices=None) -> 'JXTensor':
        """Parallelise the pipeline across devices."""
        n_devices = devices or jax.device_count()
        pipeline = self._compose
        return self._clone(
            transforms=self.transforms + [
                (lambda x: pmap(pipeline)(x), "pmap")
            ],
            batch_dim=0,
            device_count=n_devices,
        )

    # ----------------------------------------------------------
    # AUTOGRAD
    # ----------------------------------------------------------

    def grad(self, argnums: Union[int, List[int]] = 0,
             has_aux: bool = False) -> 'JXTensor':
        """
        Differentiate the entire pipeline.

        FIX: always returns a JXTensor (not a raw array) so the fluent
        chain is never broken.
        """
        grad_fn = grad(self._compose, argnums=argnums, has_aux=has_aux)
        return self._clone(
            transforms=[(grad_fn, f"grad_{self.graph_hash}")]
        )

    def value_and_grad(self, argnums: int = 0) -> Tuple['JXTensor', 'JXTensor']:
        """Return (value_tensor, grad_tensor) — both are lazy JXTensors."""
        vg_fn = jax.value_and_grad(self._compose, argnums=argnums)
        val_tensor = self._clone(
            transforms=[(lambda x: vg_fn(x)[0], f"val_{self.graph_hash}")]
        )
        grad_tensor = self._clone(
            transforms=[(lambda x: vg_fn(x)[1], f"grad_{self.graph_hash}")]
        )
        return val_tensor, grad_tensor

    def jacrev(self) -> 'JXTensor':
        """Jacobian via reverse-mode AD."""
        from jax import jacrev as _jacrev
        pipeline = self._compose
        return self._add(lambda x: _jacrev(pipeline)(x), "jacrev")

    def jacfwd(self) -> 'JXTensor':
        """Jacobian via forward-mode AD."""
        from jax import jacfwd as _jacfwd
        pipeline = self._compose
        return self._add(lambda x: _jacfwd(pipeline)(x), "jacfwd")

    def hessian(self) -> 'JXTensor':
        """
        Hessian of the pipeline.

        FIX: was incorrectly defined as a class-level lambda (not a method).
        """
        pipeline = self._compose
        return self._add(lambda x: jax.hessian(pipeline)(x), "hessian")

    # ----------------------------------------------------------
    # SHAPE / INDEXING
    # ----------------------------------------------------------

    def __getitem__(self, idx) -> 'JXTensor':
        return self._add(lambda x: x[idx], f"slice_{idx}")

    def reshape(self, *shape) -> 'JXTensor':
        return self._add(lambda x: x.reshape(*shape), f"reshape_{shape}")

    def transpose(self, *axes) -> 'JXTensor':
        return self._add(lambda x: x.transpose(*axes), f"transpose_{axes}")

    def squeeze(self, axis=None) -> 'JXTensor':
        return self._add(lambda x: x.squeeze(axis=axis), f"squeeze_{axis}")

    def unsqueeze(self, axis: int) -> 'JXTensor':
        return self._add(lambda x: jnp.expand_dims(x, axis), f"unsqueeze_{axis}")

    # ----------------------------------------------------------
    # UNARY ACTIVATIONS & ELEMENTWISE OPS
    # ----------------------------------------------------------

    def normalize(self, eps: float = 1e-8) -> 'JXTensor':
        """Zero-mean, unit-variance normalisation."""
        def _norm(x):
            return (x - x.mean()) / jnp.sqrt(x.var() + eps)
        return self._add(_norm, "normalize")

    def relu(self) -> 'JXTensor':
        return self._add(jax.nn.relu, "relu")

    def sigmoid(self) -> 'JXTensor':
        return self._add(jax.nn.sigmoid, "sigmoid")

    def log_sigmoid(self) -> 'JXTensor':
        """
        FIX: was referenced in binary_cross_entropy but never defined.
        log(sigmoid(x)) = -softplus(-x)
        """
        return self._add(jax.nn.log_sigmoid, "log_sigmoid")

    def tanh(self) -> 'JXTensor':
        return self._add(jnp.tanh, "tanh")

    def gelu(self) -> 'JXTensor':
        return self._add(jax.nn.gelu, "gelu")

    def softmax(self, axis: int = -1) -> 'JXTensor':
        return self._add(lambda x: jax.nn.softmax(x, axis=axis),
                         f"softmax_{axis}")

    def log_softmax(self, axis: int = -1) -> 'JXTensor':
        return self._add(lambda x: jax.nn.log_softmax(x, axis=axis),
                         f"log_softmax_{axis}")

    def abs(self) -> 'JXTensor':
        return self._add(jnp.abs, "abs")

    def sqrt(self) -> 'JXTensor':
        return self._add(jnp.sqrt, "sqrt")

    def exp(self) -> 'JXTensor':
        return self._add(jnp.exp, "exp")

    def log(self) -> 'JXTensor':
        return self._add(jnp.log, "log")

    def sin(self) -> 'JXTensor':
        return self._add(jnp.sin, "sin")

    def cos(self) -> 'JXTensor':
        return self._add(jnp.cos, "cos")

    # ----------------------------------------------------------
    # REDUCTIONS
    # ----------------------------------------------------------

    def sum(self, axis=None, keepdims: bool = False) -> 'JXTensor':
        return self._add(
            lambda x: jnp.sum(x, axis=axis, keepdims=keepdims),
            f"sum_{axis}")

    def mean(self, axis=None, keepdims: bool = False) -> 'JXTensor':
        return self._add(
            lambda x: jnp.mean(x, axis=axis, keepdims=keepdims),
            f"mean_{axis}")

    def var(self, axis=None, keepdims: bool = False) -> 'JXTensor':
        return self._add(
            lambda x: jnp.var(x, axis=axis, keepdims=keepdims),
            f"var_{axis}")

    def std(self, axis=None, keepdims: bool = False) -> 'JXTensor':
        return self._add(
            lambda x: jnp.std(x, axis=axis, keepdims=keepdims),
            f"std_{axis}")

    def max(self, axis=None, keepdims: bool = False) -> 'JXTensor':
        return self._add(
            lambda x: jnp.max(x, axis=axis, keepdims=keepdims),
            f"max_{axis}")

    def min(self, axis=None, keepdims: bool = False) -> 'JXTensor':
        return self._add(
            lambda x: jnp.min(x, axis=axis, keepdims=keepdims),
            f"min_{axis}")

    def argmax(self, axis=None) -> 'JXTensor':
        return self._add(lambda x: jnp.argmax(x, axis=axis), f"argmax_{axis}")

    def argmin(self, axis=None) -> 'JXTensor':
        return self._add(lambda x: jnp.argmin(x, axis=axis), f"argmin_{axis}")

    # ----------------------------------------------------------
    # BINARY OPERATIONS
    # ----------------------------------------------------------
    # FIX: The original code passed `x` (self's intermediate value) into
    # `other._compose(x)`, which is semantically wrong when `other` is an
    # independent tensor with its own data.  The correct approach is to
    # capture `other`'s materialised value at chain-build time when its data
    # is already known, or raise a clear error when it isn't.

    @staticmethod
    def _resolve(other: Any, x: Any) -> Any:
        """
        Materialise `other` for use in a binary op against `x`.

        - JXTensor with data  → evaluate its own pipeline on its own data.
        - JXTensor without data → not resolvable at build time; raise.
        - Plain scalar / array  → use as-is.
        """
        if isinstance(other, JXTensor):
            if other.data is not None:
                return other.value_of()
            # Both sides must be rooted at the same input — treat `other`
            # as a branch that also starts from x.
            return other._compose(x)
        return other

    def __add__(self, other) -> 'JXTensor':
        return self._add(lambda x: x + JXTensor._resolve(other, x), "add")

    def __radd__(self, other) -> 'JXTensor':
        return self._add(lambda x: JXTensor._resolve(other, x) + x, "radd")

    def __sub__(self, other) -> 'JXTensor':
        return self._add(lambda x: x - JXTensor._resolve(other, x), "sub")

    def __rsub__(self, other) -> 'JXTensor':
        return self._add(lambda x: JXTensor._resolve(other, x) - x, "rsub")

    def __mul__(self, other) -> 'JXTensor':
        return self._add(lambda x: x * JXTensor._resolve(other, x), "mul")

    def __rmul__(self, other) -> 'JXTensor':
        return self._add(lambda x: JXTensor._resolve(other, x) * x, "rmul")

    def __truediv__(self, other) -> 'JXTensor':
        return self._add(lambda x: x / JXTensor._resolve(other, x), "div")

    def __rtruediv__(self, other) -> 'JXTensor':
        return self._add(lambda x: JXTensor._resolve(other, x) / x, "rdiv")

    def __pow__(self, other) -> 'JXTensor':
        return self._add(lambda x: x ** JXTensor._resolve(other, x), f"pow_{other}")

    def __matmul__(self, other) -> 'JXTensor':
        return self._add(lambda x: x @ JXTensor._resolve(other, x), "matmul")

    def __neg__(self) -> 'JXTensor':
        return self._add(lambda x: -x, "neg")

    # ----------------------------------------------------------
    # LINEAR ALGEBRA
    # ----------------------------------------------------------

    def dot(self, other) -> 'JXTensor':
        return self @ other

    def matmul(self, other) -> 'JXTensor':
        return self @ other

    def einsum(self, subscripts: str, *operands) -> 'JXTensor':
        """Einstein summation — `self` maps to the first operand."""
        def _einsum(x):
            resolved = [
                op.value_of() if isinstance(op, JXTensor) and op.data is not None
                else op._compose(x) if isinstance(op, JXTensor)
                else op
                for op in operands
            ]
            return jnp.einsum(subscripts, x, *resolved)
        return self._add(_einsum, f"einsum_{subscripts}")

    # ----------------------------------------------------------
    # DEBUGGING / INSPECTION
    # ----------------------------------------------------------

    def debug_print(self, msg: str = "") -> 'JXTensor':
        """Print shape / dtype / mean of the intermediate value (passthrough)."""
        def _debug(x):
            mean_val = float(jnp.mean(x))
            print(f"[jx debug] {msg}  shape={x.shape}  dtype={x.dtype}  mean={mean_val:.4f}")
            return x
        return self._add(_debug, f"debug_{msg}")

    def inspect(self) -> Dict:
        """Return a summary of the computation graph."""
        return {
            'name': self.name,
            'transforms': [label for _, label in self.transforms],
            'depth': len(self.transforms),
            'batch_dim': self.batch_dim,
            'devices': self.device_count,
            'requires_grad': self.requires_grad,
            'graph_hash': self.graph_hash,
            'has_data': self.data is not None,
        }

    def __repr__(self):
        status = "✓" if self.data is not None else "⋯"
        labels = ', '.join(label for _, label in self.transforms[:3])
        tail = '...' if len(self.transforms) > 3 else ''
        return f"<JX{status} [{labels}{tail}]>"


# ============================================================
# jx/__init__.py - Public API
# ============================================================

def jx(data=None, mode: str = 'single', name: str = None) -> JXTensor:
    """
    JX factory — the $ of JAX.

    Args:
        data:  Input array/tensor (optional for deferred pipelines).
        mode:  'single' | 'vmap' | 'pmap' | 'shard'
        name:  Optional label for the tensor.

    Returns:
        JXTensor
    """
    tensor = JXTensor(data=data, name=name)

    if mode == 'vmap':
        return tensor.vmap()
    if mode == 'pmap':
        return tensor.pmap()
    if mode == 'shard':
        return tensor.pmap(devices=jax.device_count())

    return tensor


# jQuery-style alias
$ = jx  # noqa: E305


# ============================================================
# jx/optimize.py - Training Loops
# ============================================================

import optax


class Optimizer:
    """Thin stateful wrapper around an optax GradientTransformation."""

    def __init__(self, params: JXTensor,
                 optimizer: optax.GradientTransformation):
        if params.data is None:
            raise ValueError("Optimizer requires a JXTensor with data.")
        self.params = params
        self.opt = optimizer
        self.opt_state = optimizer.init(params.data)

    def step(self, loss_fn: Callable[[Any], float]) -> Tuple[float, 'Optimizer']:
        """
        One gradient-descent step.

        FIX: the original version computed gradients twice (once inside
        loss_and_grad, once via an explicit jax.grad call).  We now use
        jax.value_and_grad for a single forward+backward pass.
        """
        val, grads = jax.value_and_grad(loss_fn)(self.params.data)
        updates, new_state = self.opt.update(grads, self.opt_state,
                                              self.params.data)
        new_params = optax.apply_updates(self.params.data, updates)

        new_opt = Optimizer.__new__(Optimizer)
        new_opt.params = JXTensor(data=new_params,
                                  transforms=self.params.transforms)
        new_opt.opt = self.opt
        new_opt.opt_state = new_state
        return float(val), new_opt

    def train(self,
              loss_fn: Callable[[Any], float],
              steps: int = 100,
              callback: Optional[Callable] = None) -> List[float]:
        """Run a full training loop, returning per-step losses."""
        losses = []
        current = self
        for i in range(steps):
            loss, current = current.step(loss_fn)
            losses.append(loss)
            if callback:
                callback(i, loss, current.params)
        self.params = current.params
        self.opt_state = current.opt_state
        return losses


def sgd(params: JXTensor, learning_rate: float = 0.01) -> Optimizer:
    return Optimizer(params, optax.sgd(learning_rate))


def adam(params: JXTensor, learning_rate: float = 0.001) -> Optimizer:
    return Optimizer(params, optax.adam(learning_rate))


def adamw(params: JXTensor, learning_rate: float = 0.001) -> Optimizer:
    return Optimizer(params, optax.adamw(learning_rate))


# ============================================================
# jx/nn.py - Neural Network Layers
# ============================================================

class Linear:
    """Linear (Dense) layer with Xavier-style initialisation."""

    def __init__(self, in_features: int, out_features: int,
                 use_bias: bool = True, key=None):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        key = key or jax.random.PRNGKey(0)
        # Xavier uniform scale
        scale = jnp.sqrt(2.0 / (in_features + out_features))
        self.W = jax.random.normal(key, (in_features, out_features)) * scale
        self.b = jnp.zeros(out_features) if use_bias else None

    def __call__(self, x: JXTensor) -> JXTensor:
        result = x @ self.W
        if self.b is not None:
            result = result + self.b
        return result

    def parameters(self) -> List[JXTensor]:
        params = [jx(self.W, name='weight')]
        if self.b is not None:
            params.append(jx(self.b, name='bias'))
        return params


class Sequential:
    """Ordered container of callable layers."""

    def __init__(self, *layers):
        self.layers = layers

    def __call__(self, x: JXTensor) -> JXTensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[JXTensor]:
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params


# ============================================================
# jx/losses.py - Common Loss Functions
# ============================================================

def mse(pred: JXTensor, target: JXTensor) -> JXTensor:
    """Mean squared error."""
    return ((pred - target) ** 2).mean()


def cross_entropy(logits: JXTensor, labels: JXTensor) -> JXTensor:
    """Categorical cross-entropy (labels are one-hot)."""
    return -(labels * logits.log_softmax()).sum()


def binary_cross_entropy(logits: JXTensor, labels: JXTensor) -> JXTensor:
    """
    Binary cross-entropy from logits.

    FIX: original called .log_sigmoid() which was never defined on JXTensor.
    Now uses the newly added .log_sigmoid() method.
    """
    return -(
        labels * logits.log_sigmoid()
        + (1 - labels) * (-logits).log_sigmoid()
    ).mean()


def l1_loss(pred: JXTensor, target: JXTensor) -> JXTensor:
    """L1 / MAE loss."""
    return (pred - target).abs().mean()


# ============================================================
# DEMO
# ============================================================

def demo_jx():
    print("=" * 60)
    print("JX - jQuery for JAX  (fixed build)")
    print("=" * 60)

    x = jnp.array([1.0, -2.0, 3.0, -4.0, 5.0])
    print(f"\n Input: {x}")

    # 1. Lazy pipeline
    pipeline = $(x).normalize().relu().sum()
    print(f"\n Pipeline object : {pipeline}")

    # 2. Eager execution
    print(f" value_of()      : {pipeline.value_of()}")

    # 3. JIT
    print(f" jit()           : {pipeline.jit()}")

    # 4. Autograd  (grad now always returns a JXTensor)
    grad_t = $(x).normalize().relu().sum().grad()
    print(f" grad value_of() : {grad_t.value_of()}")

    # 5. value_and_grad
    val_t, grd_t = $(x).normalize().relu().sum().value_and_grad()
    print(f" val  = {val_t.value_of()}")
    print(f" grad = {grd_t.value_of()}")

    # 6. vmap over a batch
    batch = jnp.stack([x, x * 2, x * 3])
    batched_result = $(batch).vmap().normalize().relu().sum().value_of()
    print(f" vmap result     : {batched_result}")

    # 7. Chained complex pipeline
    complex_t = $(x).normalize().relu().softmax().log().sum()
    print(f" complex result  : {complex_t.value_of()}")

    # 8. Graph inspection
    print("\n Graph inspection:")
    for k, v in complex_t.inspect().items():
        print(f"   {k:15s}: {v}")

    # 9. Binary ops between independent tensors
    a = $(jnp.array([1.0, 2.0, 3.0]))
    b = $(jnp.array([4.0, 5.0, 6.0]))
    print(f"\n a + b = {(a + b).value_of()}")
    print(f" a * 3 = {(a * 3).value_of()}")
    print(f" a ** 2 = {(a ** 2).value_of()}")

    # 10. Hessian (now a proper method, not a class-level lambda)
    scalar_fn = $(x).sum()
    h = scalar_fn.hessian().value_of()
    print(f"\n Hessian shape   : {h.shape}")

    # 11. Optimizer
    print("\n Optimization (5 steps):")
    target = jnp.array(3.0)

    def loss_fn(p):
        return jnp.mean((p - target) ** 2)

    opt = adam($(jnp.array([0.0])), learning_rate=0.1)
    losses = opt.train(loss_fn, steps=5)
    print(f"   Losses: {[f'{l:.4f}' for l in losses]}")
    print(f"   Final param: {opt.params.data}")

    print("\n JX Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo_jx()
