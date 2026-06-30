"""
Tests for the delta / delta-delta MFCC feature computation in
nps_acoustic_discovery.feature.FeatureExtractor.

These only need numpy + python_speech_features (the feature path), not keras
or the binary models that test_model.py depends on.
"""

import numpy as np

from nps_acoustic_discovery.feature import FeatureExtractor


FCONFIG = {
    'hop_size': 0.01,
    'num_cepstral_coeffs': 13,
    'num_filters': 26,
    'nfft': 512,
    'low_freq': 0,
    'high_freq': None,
}


def _reference_regression(coeffs, N=2):
    """
    Independent textbook delta computation on the interior frames:

        delta[t] = sum_{n=1..N} n * (c[t+n] - c[t-n]) / (2 * sum_{n=1..N} n^2)

    Implemented per-coefficient so it can't accidentally share the bug.
    Boundary frames (where t +/- N runs off either edge) are left as NaN and
    skipped by the caller, since this test is about the regression math, not
    the existing edge fallback.
    """
    num_frames, num_coeffs = coeffs.shape
    denom = 2.0 * sum(n ** 2 for n in range(1, N + 1))
    out = np.full_like(coeffs, np.nan, dtype=float)
    for t in range(N, num_frames - N):
        for c in range(num_coeffs):
            numer = sum(n * (coeffs[t + n, c] - coeffs[t - n, c]) for n in range(1, N + 1))
            out[t, c] = numer / denom
    return out


def _make_audio():
    # Deterministic synthetic signal so the test is reproducible.
    sample_rate = 16000
    rng = np.random.RandomState(0)
    t = np.linspace(0, 1.0, sample_rate, endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    noise = 0.05 * rng.randn(sample_rate)
    return (tone + noise).astype(np.float64), sample_rate


def _split_feature_blocks(features, num_cep):
    mfcc_feat = features[:, :num_cep]
    mfcc_delta = features[:, num_cep:2 * num_cep]
    mfcc_delta_delta = features[:, 2 * num_cep:]
    return mfcc_feat, mfcc_delta, mfcc_delta_delta


def test_delta_features_are_per_coefficient_not_collapsed():
    """A frame's delta values must differ across cepstral coefficients.

    The old code summed without axis=0, collapsing every coefficient in a
    frame to a single repeated scalar.
    """
    audio, sr = _make_audio()
    features = FeatureExtractor(FCONFIG).process(audio, sr)
    _, mfcc_delta, mfcc_delta_delta = _split_feature_blocks(features, FCONFIG['num_cepstral_coeffs'])

    # Look at an interior frame (boundary frames fall back to the raw coeffs).
    interior = mfcc_delta[10, :]
    assert np.unique(np.round(interior, 8)).size > 1, \
        "delta coefficients within a frame collapsed to one repeated value"

    interior_dd = mfcc_delta_delta[10, :]
    assert np.unique(np.round(interior_dd, 8)).size > 1, \
        "delta-delta coefficients within a frame collapsed to one repeated value"


def test_delta_matches_independent_reference():
    """The delta/delta-delta blocks must match the textbook regression formula.

    This also pins the normalization denominator (2 * sum n^2 over 1..N),
    which the old range(N) version got wrong by an off-by-one.
    """
    audio, sr = _make_audio()
    features = FeatureExtractor(FCONFIG).process(audio, sr)
    mfcc_feat, mfcc_delta, mfcc_delta_delta = _split_feature_blocks(
        features, FCONFIG['num_cepstral_coeffs'])

    expected_delta = _reference_regression(mfcc_feat)
    expected_delta_delta = _reference_regression(mfcc_delta)

    # Compare only the interior frames the reference filled in (NaN elsewhere).
    delta_interior = ~np.isnan(expected_delta)
    dd_interior = ~np.isnan(expected_delta_delta)
    assert delta_interior.any() and dd_interior.any()

    assert np.allclose(mfcc_delta[delta_interior], expected_delta[delta_interior]), \
        "delta MFCC block does not match the reference regression formula"
    assert np.allclose(mfcc_delta_delta[dd_interior], expected_delta_delta[dd_interior]), \
        "delta-delta MFCC block does not match the reference regression formula"
