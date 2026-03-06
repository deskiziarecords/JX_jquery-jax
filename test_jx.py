import jax
import jax.numpy as jnp
from JX import jq

def test_value_of():
    x = jnp.array([1.0, 2.0, 3.0])
    res = jq(x).sum().value_of()
    assert res == 6.0
    print("test_value_of passed")

def test_vmap_alias():
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    res = jq(x).vmap().sum().value()
    assert jnp.all(res == jnp.array([3.0, 7.0]))
    print("test_vmap_alias passed")

def test_pmap():
    try:
        num_devices = jax.local_device_count()
        if num_devices > 0:
            x_pmap = jnp.ones((num_devices, 4))
            res = jq(x_pmap).pmap().sum().value()
            assert res.shape == (num_devices,)
            print(f"test_pmap passed with {num_devices} devices")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"test_pmap failed: {e}")

if __name__ == "__main__":
    test_value_of()
    test_vmap_alias()
    test_pmap()
