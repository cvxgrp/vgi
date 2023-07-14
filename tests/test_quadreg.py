import numpy as np
from vgi.quad_form import QuadForm
from vgi.quad_reg import QuadReg, QuadGradReg


def test_quadreg():
    np.random.seed(0)
    V = QuadForm.random(5)
    x = np.random.randn(500, 5)
    y = V(x)

    fitter = QuadReg(l2_penalty=1e-3, l1_penalty=1e-4).fit(x, y)
    assert np.allclose(fitter.V_.params, V.params, atol=1e-4)


def test_quadgradreg():
    np.random.seed(0)
    V = QuadForm.random(5)
    V.c = 0.0
    x = np.random.randn(500, 5)
    y = V.grad(x)

    fitter = QuadGradReg(l2_penalty=1e-4, l1_penalty=1e-4).fit(x, y)
    assert np.allclose(fitter.V_.params, V.params, atol=1e-4)
