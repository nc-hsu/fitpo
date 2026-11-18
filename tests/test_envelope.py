"""Tests for fit_envelope and its tracing modes."""

import numpy as np
import pytest

from fitpo import fit_envelope


def test_invalid_mode_raises(fracture_cyclic):
    with pytest.raises(ValueError, match="mode must be one of"):
        fit_envelope(fracture_cyclic.curve, mode="nope")  # type: ignore[arg-type]


def test_nonpositive_bin_size_raises(fracture_cyclic):
    with pytest.raises(ValueError, match="bin_size must be positive"):
        fit_envelope(fracture_cyclic.curve, mode="displacement-bins", bin_size=0.0)


def test_peak_to_peak_regression(fracture_cyclic):
    """Default peak-to-peak output: origin + each cycle tip, missing the +5 end."""
    pos, neg = fit_envelope(fracture_cyclic.curve)

    expected_pos = np.asarray([[0.0, 0.0], [1.0, 100.0], [2.0, 180.0], [3.0, 240.0], [4.0, 80.0]])
    assert pos.shape == expected_pos.shape
    assert np.allclose(pos, expected_pos, atol=1e-6)

    expected_neg = np.asarray([[0.0, 0.0], [-1.0, -100.0], [-2.0, -180.0], [-3.0, -240.0]])
    assert neg.shape == expected_neg.shape
    assert np.allclose(neg, expected_neg, atol=1e-6)


def test_envelopes_start_at_origin(fracture_cyclic):
    for mode in (
        "peak-to-peak",
        "loading-front",
        "displacement-bins",
        "adjusted-peak-to-peak",
        "simplified-loading-front",
    ):
        pos, neg = fit_envelope(fracture_cyclic.curve, mode=mode)
        assert np.allclose(pos[0], [0.0, 0.0])
        assert np.allclose(neg[0], [0.0, 0.0])
        assert np.all(pos[1:, 0] > 0.0)
        assert np.all(neg[1:, 0] < 0.0)


@pytest.mark.parametrize("mode", ["loading-front", "displacement-bins"])
def test_outer_modes_capture_true_peak(fracture_cyclic, mode):
    """Outer-trace modes reach the 250 peak between tips; peak-to-peak caps at 240."""
    peak_pos, _ = fit_envelope(fracture_cyclic.curve)
    pos, _ = fit_envelope(fracture_cyclic.curve, mode=mode)

    assert pos[:, 1].max() > peak_pos[:, 1].max()
    assert pos[:, 1].max() == pytest.approx(250.0, abs=2.0)


@pytest.mark.parametrize("mode", ["loading-front", "displacement-bins"])
def test_outer_modes_dip_after_fracture(fracture_cyclic, mode):
    """After the peak-force breakpoint the force drops well below it (fracture)."""
    pos, _ = fit_envelope(fracture_cyclic.curve, mode=mode)
    i_peak = int(np.argmax(pos[:, 1]))
    after = pos[i_peak + 1 :, 1]
    assert after.size > 0
    assert after.min() < 0.5 * pos[i_peak, 1]


def test_adjusted_captures_peak_with_fewer_points(fracture_cyclic):
    """adjusted-p2p reaches the true 250 peak but stays leaner than loading-front."""
    p2p, _ = fit_envelope(fracture_cyclic.curve, mode="peak-to-peak")
    adj, _ = fit_envelope(fracture_cyclic.curve, mode="adjusted-peak-to-peak")
    front, _ = fit_envelope(fracture_cyclic.curve, mode="loading-front")

    assert adj[:, 1].max() > p2p[:, 1].max()
    assert adj[:, 1].max() == pytest.approx(250.0, abs=2.0)
    # inserts the apex, so more points than plain p2p, but far fewer than the
    # full loading front (saw teeth dropped).
    assert adj.shape[0] > p2p.shape[0]
    assert adj.shape[0] < front.shape[0]


def test_adjusted_drops_through_fracture(fracture_cyclic):
    """After the inserted peak the force drops well below it (fracture captured)."""
    adj, _ = fit_envelope(fracture_cyclic.curve, mode="adjusted-peak-to-peak")
    i_peak = int(np.argmax(adj[:, 1]))
    after = adj[i_peak + 1 :, 1]
    assert after.size > 0
    assert after.min() < 0.5 * adj[i_peak, 1]


def test_adjusted_reduces_to_p2p_without_protrusion():
    """With a convex loading branch (all points below the chord), adjusted == p2p."""
    d_up = np.linspace(0.0, 5.0, 120)
    f_up = 100.0 * (d_up / 5.0) ** 2  # convex: lies below the (0,0)->(5,100) chord
    d_down = np.linspace(5.0, 4.0, 30)[1:]
    f_down = np.linspace(100.0, 60.0, 30)[1:]
    curve = np.column_stack([np.concatenate([d_up, d_down]), np.concatenate([f_up, f_down])])

    p2p, _ = fit_envelope(curve, mode="peak-to-peak")
    adj, _ = fit_envelope(curve, mode="adjusted-peak-to-peak")
    assert adj.shape == p2p.shape
    assert np.allclose(adj, p2p, atol=1e-6)


def test_simplified_front_captures_peak_with_fewer_points(fracture_cyclic):
    """simplified-loading-front reaches the true 250 apex but stays leaner than loading-front.

    The apex (250) is only ~4% above the 240 tip, so the default 5% tolerance
    smooths it; a tighter tolerance treats it as significant and captures it.
    """
    p2p, _ = fit_envelope(fracture_cyclic.curve, mode="peak-to-peak")
    ref, _ = fit_envelope(
        fracture_cyclic.curve, mode="simplified-loading-front", simplify_tolerance=0.02
    )
    front, _ = fit_envelope(fracture_cyclic.curve, mode="loading-front")

    assert ref[:, 1].max() > p2p[:, 1].max()
    assert ref[:, 1].max() == pytest.approx(250.0, abs=3.0)
    assert ref.shape[0] > p2p.shape[0]
    assert ref.shape[0] < front.shape[0]


def test_simplified_front_drops_through_fracture(fracture_cyclic):
    """After the captured apex the force drops well below it (fracture captured)."""
    ref, _ = fit_envelope(fracture_cyclic.curve, mode="simplified-loading-front")
    i_peak = int(np.argmax(ref[:, 1]))
    after = ref[i_peak + 1 :, 1]
    assert after.size > 0
    assert after.min() < 0.5 * ref[i_peak, 1]


def test_simplified_front_reduces_to_p2p_without_deviation():
    """With a straight loading branch (no deviation), simplified-loading-front == plain p2p."""
    d_up = np.linspace(0.0, 5.0, 120)
    f_up = 100.0 * (d_up / 5.0)  # straight: no deviation from the (0,0)->(5,100) chord
    d_down = np.linspace(5.0, 4.0, 30)[1:]
    f_down = np.linspace(100.0, 60.0, 30)[1:]
    curve = np.column_stack([np.concatenate([d_up, d_down]), np.concatenate([f_up, f_down])])

    p2p, _ = fit_envelope(curve, mode="peak-to-peak")
    ref, _ = fit_envelope(curve, mode="simplified-loading-front")
    assert ref.shape == p2p.shape
    assert np.allclose(ref, p2p, atol=1e-6)


def test_simplified_front_follows_sheer_drop_below_chord(postpeak_drop_cyclic):
    """Only simplified-loading-front follows the post-peak drop; p2p and adjusted overestimate."""
    curve = postpeak_drop_cyclic.curve
    p2p, _ = fit_envelope(curve, mode="peak-to-peak")
    adj, _ = fit_envelope(curve, mode="adjusted-peak-to-peak")
    ref, _ = fit_envelope(curve, mode="simplified-loading-front")

    probe = 3.0  # between the (2, 200) and (4, 55) tips, where the chord overestimates
    p2p_at = np.interp(probe, p2p[:, 0], p2p[:, 1])
    adj_at = np.interp(probe, adj[:, 0], adj[:, 1])
    ref_at = np.interp(probe, ref[:, 0], ref[:, 1])

    assert adj_at == pytest.approx(p2p_at, abs=1.0)  # hull leaves the chord in place
    assert ref_at < 0.5 * p2p_at
    assert ref_at < adj_at


def test_include_final_peak_adds_endpoint(fracture_cyclic):
    """The +5 endpoint (global max disp) is only kept when include_final_peak."""
    without, _ = fit_envelope(fracture_cyclic.curve, mode="peak-to-peak")
    with_end, _ = fit_envelope(fracture_cyclic.curve, mode="peak-to-peak", include_final_peak=True)

    assert without[-1, 0] == pytest.approx(4.0, abs=1e-6)
    assert with_end[-1, 0] == pytest.approx(5.0, abs=1e-6)
    assert with_end.shape[0] == without.shape[0] + 1


def test_close_at_zero_drops_to_zero_force(fracture_cyclic):
    pos, neg = fit_envelope(fracture_cyclic.curve, mode="loading-front", close_at_zero=True)

    assert pos[-1, 1] == pytest.approx(0.0, abs=1e-9)
    assert pos[-1, 0] == pytest.approx(pos[-2, 0], abs=1e-9)
    assert neg[-1, 1] == pytest.approx(0.0, abs=1e-9)
    assert neg[-1, 0] == pytest.approx(neg[-2, 0], abs=1e-9)
