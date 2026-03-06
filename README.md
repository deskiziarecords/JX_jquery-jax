# YAKJAX
**Chainable, lazy tensor ops for JAX – because fuck boilerplate.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.0+-orange.svg)](https://github.com/google/jax)

> Built because I needed simple chains without the ceremony. Yaks don't do boilerplate.

![YAKJAX mascot – geometric yak with JAX branding](https://github.com/deskiziarecords/YAKJAX/blob/main/YAKJAX.jpg)  
*(shaggy, sturdy, no nonsense – just like the API)*

## The Pain

Raw JAX is the best invention since the electric motor. Pure. Powerful.  
But chaining five ops turns into nested hell and even looks kind of grotesque (in my opinion), a basicathrowaway `@jit` function nobody wants to read twice.

You write:
```python
y = jnp.dot(x, W) + b
h = jax.nn.relu(y)
z = jax.nn.softmax(jnp.dot(h, W2) + b2)
```

You deserve better. Left-to-right. No nesting. No mandatory closures.

## The Fix

```python
from yakjax import YakTensor

z = (YakTensor(x)
     .matmul(W)
     .add(b)
     .relu()
     .matmul(W2)
     .add(b2)
     .softmax()
     .jit())          # compile when you say so
```

That's it. Read like a sentence. Lazy but solid as a YAK – nothing computes until you pull the value.

## Quick & Dirty

```python
import jax.numpy as jnp
from yakjax import YakTensor, Linear

x = jnp.ones((32, 784))

# Build once, execute however
pipeline = (YakTensor(x)
    .matmul(jnp.ones((784, 256)))
    .add(jnp.ones(256))
    .relu()
    .matmul(jnp.ones((256, 10)))
    .softmax())

result     = pipeline.value_of()     # run now
compiled   = pipeline.jit()          # XLA'd beast mode
batched    = pipeline.vmap(0)        # vectorize over batch dim
```

## Lazy = Your Friend

Build graph → inspect → execute. Fusion happens automatically when jitted.

```python
graph = (YakTensor(x)
    .reshape(100, 28, 28)
    .transpose(0, 2, 1)
    .dot(kernel))

print(graph.inspect())   # ops: 3, shape: (100,28,28), dtype: float32

result = graph.value_of()  # now it runs
```

## Gradients, No Tears

```python
loss = (YakTensor(x)
    .matmul(W1).relu()
    .matmul(W2)
    .mse(target))

grads = loss.grad([W1, W2])               # just the grads
val, grads = loss.value_and_grad([W1, W2]) # value + grads
```

No `jax.grad(fn)`, no arg threading, when you step over a function and it feels like stepping over a fresh YAK's crap.

## Higher-Level 

```python
from yakjax import Sequential, Linear, adam

model = Sequential([
    Linear(784, 256), 'relu',
    Linear(256, 128), 'relu',
    Linear(128, 10)
])

logits = YakTensor(batch).sequential(model)
loss   = logits.cross_entropy(labels)

opt    = adam(1e-3)
params = opt.step(loss, model.parameters())
```

## When to use or ride the YAK

**Use it if:**
- Prototyping / experiments / teaching
- You hate nested calls and function spam ( Still finding a cure ...)
- Small-to-medium models where readability > micro-optimizations

**Skip it if:**
- Production at scale (pjit, shard_map, custom XLA hell)
- You need absolute control over every compilation detail
- You're allergic to any wrapper, even a thin one

This ain't replacing Flax / Haiku / Equinox. It's just riding a YAk to enjoy the surroundings .

## Install

```bash
pip install yakjax
```

Requires: Python ≥3.9, JAX ≥0.4.0

## Under the Hood

Plain JAX underneath. No custom kernels. Just composition + lazy graph.  
Explicit `.jit()` – no auto-magic surprises.  
If it breaks your advanced vjp/pmap/donate shit, file an issue. I'll fix or document why not.

## Design Notes
(Even if it's out of scoupe)
I admit that I never been fully recovered since the day I took a peek inside a nuxt-app empty folder with a 400MB node_modules.  
I faced nest mental attack !!! that "nest-js" spider/nesting/nest/nesting/squizofrenic/nested/nests caused me some shock.

## Contributing

 PRs welcome, especially:
- Missing chainable ops that make sense
- Lazy eval edge cases that explode
- Perf comparisons vs raw JAX

Issues: "this YAK crapped my custom grad" → gold.  
"add .conv" → also gold.

**Author:** Roberto Jiménez (@hipotermiah)  
**License:** MIT

Yaks carry the load. You chain the ops. Enjoy.
```

 🐂
