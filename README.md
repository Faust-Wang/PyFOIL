# PyFOIL

[![Build Status](https://travis-ci.org/danielkelshaw/PyFOIL.svg?branch=master)](https://travis-ci.org/danielkelshaw/PyFOIL)

Library for aerofoil parameterisations

- [x] MIT License
- [x] Python 3.6+

## **Example:**

In order to utilise a parameterisation you simply need to supply the
class with the relevant parameters can call the `generate()` method - an
example of this for the BP3333 parameterisation technique can be seen
below: 

```python
import numpy as np
from pyfoil.bezier_parsec import BP3333


while True:
    
    t_params = {
        'r_le': np.random.uniform(-0.04, -0.001),
        'x_t': np.random.uniform(0.15, 0.4),
        'y_t': np.random.uniform(0.05, 0.15),
        'k_t': np.random.uniform(-0.5, 0.0),
        'dz_te': np.random.uniform(0.0, 0.001),
        'beta_te': np.random.uniform(0.001, 0.3)
    }
    
    c_params = {
        'gamma_le': np.random.uniform(0.05, 0.1),
        'x_c': np.random.uniform(0.2, 0.5),
        'y_c': np.random.uniform(0.0, 0.2),
        'k_c': np.random.uniform(-0.2, -0.0),
        'z_te': np.random.uniform(0.0, 0.01),
        'alpha_te': np.random.uniform(0.05, 0.1)
    }
    
    params = {**t_params, **c_params}
    model = BP3333(params)

    if not model.violates_constraints:
        break
        
x_u, y_u, x_l, y_l = model.generate()
```

###### Made by Daniel Kelshaw
