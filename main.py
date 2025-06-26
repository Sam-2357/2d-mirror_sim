"""
─────────────────────────────
Complete reference implementation for **one 2-D Optotune MR-E-2 mirror**
and a single laser emitter, with ±25° steering and 1.3 mm standoff.

Author : Sam Muddimer
Date   : 26 Jun 2025
"""

# ───────────────────────────────────────────────────────────────
# 0. Imports
# ───────────────────────────────────────────────────────────────
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go

# ───────────────────────────────────────────────────────────────
# 1. Global constants / defaults
# ───────────────────────────────────────────────────────────────
MAX_TILT_DEG = 25.0
MAX_TILT_RAD = np.deg2rad(MAX_TILT_DEG)

D_STANDOFF = 1.3e-3          # 1.3 mm
SEG_LEN    = 50e-3           # 50 mm visible mirror surface
BEAM_LEN   = 0.500           # 0.5 m outgoing beam for display
O_PIVOT = np.array([0.0, 0.0], dtype=float)
E_EMIT  = np.array([10.0, 10.0], dtype=float)
N_REST     = np.array([1.0, 0.0], dtype=float)   # rest-state normal (+x)
THETA_INIT = 0.0                                 # initial tilt (rad)
# ───────────────────────────────────────────────────────────────
# 2. Basic maths helpers
# ───────────────────────────────────────────────────────────────
def _rot(theta: float) -> np.ndarray:
    """2-D CCW rotation matrix."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

def _unit(vec: np.ndarray) -> np.ndarray:
    """Return vec / ‖vec‖."""
    nrm = np.linalg.norm(vec)
    if nrm == 0:
        raise ValueError("Zero-length vector cannot be normalised.")
    return vec / nrm


# ───────────────────────────────────────────────────────────────
# 3. Core classes
# ───────────────────────────────────────────────────────────────
class Mirror2D:
    """Flat mirror with finite visible surface."""
    def __init__(
        self,
        pivot:   np.ndarray,
        n_rest:  np.ndarray,
        d:       float = D_STANDOFF,
        seg_L:   float = SEG_LEN,
        theta:   float = 0.0,
    ) -> None:
        self.pivot   = np.asarray(pivot, dtype=float)
        self.n_rest  = _unit(np.asarray(n_rest, dtype=float))
        self.d       = float(d)
        self.seg_L   = float(seg_L)
        self._theta  = 0.0
        self.set_tilt(theta)

    # ——— public API ———
    def set_tilt(self, theta: float) -> None:
        """Clamp θ to mechanical limits and store."""
        theta = float(theta)
        if abs(theta) > MAX_TILT_RAD:
            raise ValueError(f"|θ| must be ≤ {MAX_TILT_DEG}°")
        self._theta = theta

    # current normal
    def normal(self) -> np.ndarray:
        return _rot(self._theta) @ self.n_rest

    # surface strike point S = O + d n
    def surface_point(self) -> np.ndarray:
        return self.pivot + self.d * self.normal()

    # reflect unit vector about current normal
    def reflect(self, u_hat: np.ndarray) -> np.ndarray:
        n = self.normal()
        return _unit(u_hat - 2 * (u_hat @ n) * n)

    # endpoints of visible 50 mm mirror surface
    def surface_segment(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.normal()
        tangent = np.array([-n[1], n[0]])
        S = self.surface_point()
        return (S - 0.5 * self.seg_L * tangent,
                S + 0.5 * self.seg_L * tangent)


class Emitter2D:
    """Point laser emitter with fixed origin and heading ψ."""
    def __init__(self, pos: np.ndarray, angle: float) -> None:
        self.pos   = np.asarray(pos, dtype=float)
        self.angle = float(angle)

    def direction(self) -> np.ndarray:
        return np.array([np.cos(self.angle), np.sin(self.angle)])


# ───────────────────────────────────────────────────────────────
# 4. Stand-alone utilities
# ───────────────────────────────────────────────────────────────
def intersect_line_with_segment(
    P0: np.ndarray, d0: np.ndarray,
    S1: np.ndarray, S2: np.ndarray
) -> np.ndarray | None:
    """
    Intersection point between the infinite line P0+λd0 (λ ∈ ℝ)
    and the finite segment [S1,S2]. Returns None if no hit.
    """
    d0 = np.asarray(d0, dtype=float)
    S1, S2 = map(np.asarray, (S1, S2))
    v = S2 - S1
    M = np.column_stack((d0, -v))
    b = S1 - P0
    if abs(np.linalg.det(M)) < 1e-12:
        return None                     # parallel
    lam, mu = np.linalg.solve(M, b)
    if 0.0 <= mu <= 1.0:                # within the segment
        return P0 + lam * d0
    return None


def standoff_correction(alpha: float, d: float = D_STANDOFF) -> float:
    """Δℓ = 2 d sin(α/2)."""
    return 2.0 * d * np.sin(alpha / 2.0)


# ───────────────────────────────────────────────────────────────
# 5. Plotly figure generator
# ───────────────────────────────────────────────────────────────
def build_plotly_figure(
    emitter: Emitter2D,
    mirror:  Mirror2D,
    beam_len: float = BEAM_LEN,
) -> go.Figure:
    """Return a fully annotated Plotly Figure."""
    # 5.1 incoming beam (infinite line from Ε along û)
    u_hat = _unit(mirror.surface_point() - emitter.pos)
    S1, S2 = mirror.surface_segment()
    P_hit = intersect_line_with_segment(emitter.pos, u_hat, S1, S2)
    if P_hit is None:
        raise RuntimeError("Incoming beam misses mirror segment.")

    # 5.2 outgoing beam
    v_hat = mirror.reflect(u_hat)

    # 5.3 angles
    alpha = np.arccos(np.clip(u_hat @ mirror.normal(), -1.0, 1.0))
    delta_L = standoff_correction(alpha)

    # 5.4 build figure
    fig = go.Figure()

    # incoming beam
    fig.add_scatter(
        x=[emitter.pos[0], P_hit[0]],
        y=[emitter.pos[1], P_hit[1]],
        mode="lines",
        line=dict(color="black"),
        name="incoming"
    )
    # mirror surface
    fig.add_scatter(
        x=[S1[0], S2[0]],
        y=[S1[1], S2[1]],
        mode="lines",
        line=dict(color="orange", width=4),
        name="mirror surface"
    )
    # normal
    n_end = mirror.surface_point() + 0.04 * mirror.normal()  # 40 mm arrow
    fig.add_scatter(
        x=[mirror.surface_point()[0], n_end[0]],
        y=[mirror.surface_point()[1], n_end[1]],
        mode="lines+markers",
        line=dict(color="blue"),
        marker=dict(size=6, color="blue"),
        name="normal"
    )
    # standoff vector
    fig.add_scatter(
        x=[mirror.pivot[0], mirror.surface_point()[0]],
        y=[mirror.pivot[1], mirror.surface_point()[1]],
        mode="lines+markers",
        line=dict(color="red", dash="dot"),
        marker=dict(size=6, color="red"),
        name="standoff d"
    )
    # outgoing beam
    P_out = P_hit + beam_len * v_hat
    fig.add_scatter(
        x=[P_hit[0], P_out[0]],
        y=[P_hit[1], P_out[1]],
        mode="lines",
        line=dict(color="green"),
        name="outgoing"
    )

    # annotations
    fig.add_annotation(
        x=P_hit[0], y=P_hit[1],
        text=f"α = {np.degrees(alpha):.2f}°<br>Δℓ = {delta_L*1e3:.3f} mm",
        showarrow=True, arrowhead=1, ax=40, ay=-40
    )
    fig.update_layout(
        width=700, height=700,
        xaxis=dict(scaleanchor="y"), showlegend=True,
        title="2-D Optotune mirror geometry"
    )
    return fig


# ───────────────────────────────────────────────────────────────
# 6. Convenience “demo” entry point
# ───────────────────────────────────────────────────────────────
def _demo() -> None:
    """Run a quick sanity-check figure with default parameters."""
    emitter = Emitter2D(E_EMIT, np.arctan2(O_PIVOT[1]-E_EMIT[1],
                                          O_PIVOT[0]-E_EMIT[0]))
    mirror  = Mirror2D(O_PIVOT, N_REST, theta=THETA_INIT)
    fig = build_plotly_figure(emitter, mirror)
    fig.show()


# Allow “python -m mirror_viz.model2d_outline” to launch the demo
if __name__ == "__main__":
    _demo()
