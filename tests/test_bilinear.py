"""Tests for fit_bilinear_backbone."""

import numpy as np

from fitpo import fit_bilinear_backbone


def test_bilinear(bilinear_source):
    fit = fit_bilinear_backbone(bilinear_source.curve, mode="bilinear-pctloss").fitted_po
    assert fit.shape == (3, 2)
    assert np.allclose(fit[0], [0.0, 0.0])
    d_y, f_y = fit[1]
    d_end, _f_end = fit[2]
    assert d_y > 0 and f_y > 0
    assert d_end > d_y
