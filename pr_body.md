## Summary
Add MPS (Metal Performance Shaders) support for the `heaviside` operation on Apple Silicon.

## Description
This PR implements the Heaviside step function for the MPS backend using MPSGraph's `selectWithPredicateTensor` operations.

The heaviside function is defined as:
- `heaviside(a, b) = 0` if `a < 0`
- `heaviside(a, b) = b` if `a == 0`
- `heaviside(a, b) = 1` if `a > 0`

## Changes
- Add `heaviside_out_mps` implementation in `aten/src/ATen/native/mps/operations/BinaryOps.mm`
- Add MPS dispatch entry in `native_functions.yaml`
- Add comprehensive tests in `test/test_mps.py` covering various dtypes (float32, float16, bfloat16) and shapes including broadcasting

## Test Plan
```python
import torch

a = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], device='mps')
b = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5], device='mps')
print(torch.heaviside(a, b))
# Expected: tensor([0., 0., 0.5, 1., 1.], device='mps:0')
```

cc @malfet @kulinseth
