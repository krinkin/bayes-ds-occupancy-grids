"""Non-homogeneous inverse sensor model for H-002 experiment.

Provides distance- and angle-dependent log-odds and DS/TBM mass function
values for occupancy grid updates.  Unlike H-003's constant sensor model,
update values vary per cell based on distance from the robot and angle of
incidence, breaking the (L,n) <-> (m_O, m_F, m_OF) bijection.

Two matching strategies are available:

BetP matching (default, used in all H-002 experiments):
    Masses derived so that BetP(O) = sigmoid(l).
    For p = sigmoid(l) >= 0.5:  m_O = 2p-1, m_F = 0, m_OF = 2(1-p).
    BetP(O) = m_O + m_OF/2 = (2p-1) + (1-p) = p. Exact.

PPl matching (reviewer robustness check M2):
    Masses derived so that PPl(O) = sigmoid(l).
    PPl(O) = Pl(O) / (Pl(O) + Pl(F)).
    For consonant BBAs (m_F = 0): Pl(O) = m_O + m_OF = 1, Pl(F) = m_OF.
    PPl(O) = 1 / (1 + m_OF) = p  =>  m_OF = (1-p)/p, m_O = (2p-1)/p.
    For p < 0.5: m_O = 0, Pl(F) = m_F + m_OF = 1, Pl(O) = m_OF.
    PPl(O) = m_OF / (1 + m_OF) = p => m_OF = p/(1-p), m_F = (1-2p)/(1-p).
"""

from __future__ import annotations

import math


def _betp_masses(p: float) -> tuple[float, float, float]:
    """Derive consonant DS masses such that BetP(O) = p.

    BetP matching: BetP(O) = m_O + m_OF/2 = p.

    Valid for p in [0, 1].  Returns (m_O, m_F, m_OF) summing to 1.0.
    """
    m_O = max(0.0, 2.0 * p - 1.0)
    m_F = max(0.0, 1.0 - 2.0 * p)
    m_OF = 1.0 - abs(2.0 * p - 1.0)
    return m_O, m_F, m_OF


def _ppl_masses(p: float) -> tuple[float, float, float]:
    """Derive consonant DS masses such that PPl(O) = p.

    Normalized plausibility matching: PPl(O) = Pl(O) / (Pl(O) + Pl(F)) = p.

    For p >= 0.5 (occupied evidence), m_F = 0 (consonant towards O):
        Pl(O) = m_O + m_OF = 1,  Pl(F) = m_OF
        PPl(O) = 1 / (1 + m_OF) = p  =>  m_OF = (1-p)/p, m_O = (2p-1)/p.

    For p < 0.5 (free evidence), m_O = 0 (consonant towards F):
        Pl(F) = m_F + m_OF = 1,  Pl(O) = m_OF
        PPl(O) = m_OF / (1 + m_OF) = p  =>  m_OF = p/(1-p), m_F = (1-2p)/(1-p).

    At p = 0.5: m_O = 0, m_F = 0, m_OF = 1 (vacuous), same as BetP.
    At p -> 1:  m_O -> 1, m_F = 0, m_OF -> 0.
    At p -> 0:  m_O = 0, m_F -> 1, m_OF -> 0.

    Valid for p in (0, 1).  Returns (m_O, m_F, m_OF) summing to 1.0.
    """
    eps = 1e-12
    p = max(eps, min(1.0 - eps, p))

    if p >= 0.5:
        m_OF = (1.0 - p) / p
        m_O = 1.0 - m_OF   # = (2p-1)/p
        m_F = 0.0
    else:
        m_OF = p / (1.0 - p)
        m_F = 1.0 - m_OF   # = (1-2p)/(1-p)
        m_O = 0.0

    # Numerical guard: renormalize
    total = m_O + m_F + m_OF
    if total > 1e-10:
        m_O /= total
        m_F /= total
        m_OF /= total

    return m_O, m_F, m_OF


class NonHomogeneousSensorModel:
    """Sensor model with distance- and angle-dependent update values.

    Log-odds model::

        l_occ  = base_l_occ  * exp(-distance_decay * d) * exp(-angle_decay * a)
        l_free = base_l_free * exp(-distance_decay * d) * exp(-angle_decay * a)

    Both factors apply uniformly; the caller passes ``angle=0`` for free-space
    cells (ray passes through) so that only distance modulates confidence,
    while the hit cell uses the actual angle of incidence.

    DS/TBM mass model derived from log-odds via either the pignistic transform
    (default, ``matching="betp"``) or the normalized plausibility transform
    (``matching="ppl"``).  Both ensure that the chosen transform applied to the
    derived masses equals sigmoid(l).

    Parameters
    ----------
    base_l_occ:
        Log-odds increment for an occupied hit at distance=0, angle=0.
    base_l_free:
        Log-odds update for a free-traversal cell at distance=0, angle=0.
        Typically negative (evidence for free).
    distance_decay:
        Exponential decay rate for distance (1/metres).
    angle_decay:
        Exponential decay rate for angle of incidence (1/radians).
    max_range:
        Sensor maximum range in metres (informational, not used in formulas).
    matching:
        Mass derivation strategy: ``"betp"`` (pignistic, default) or
        ``"ppl"`` (normalized plausibility).  Does not affect log-odds.
    """

    def __init__(
        self,
        base_l_occ: float = 2.0,
        base_l_free: float = -0.5,
        distance_decay: float = 0.1,
        angle_decay: float = 0.5,
        max_range: float = 10.0,
        matching: str = "betp",
    ) -> None:
        if matching not in ("betp", "ppl"):
            raise ValueError(f"matching must be 'betp' or 'ppl', got {matching!r}")
        self.base_l_occ = base_l_occ
        self.base_l_free = base_l_free
        self.distance_decay = distance_decay
        self.angle_decay = angle_decay
        self.max_range = max_range
        self.matching = matching

    def compute_log_odds(
        self,
        distance: float,
        angle_of_incidence: float,
        cell_type: str = "occupied",
    ) -> tuple[float, float]:
        """Compute distance/angle-dependent log-odds update values.

        Parameters
        ----------
        distance:
            Distance from robot to cell in metres (>= 0).
        angle_of_incidence:
            Angle of incidence in radians [0, pi/2].
            0 = perpendicular (best case), pi/2 = grazing (worst case).
        cell_type:
            ``"occupied"`` or ``"free"``.  Reserved for future use; both
            l_occ and l_free are always computed regardless of this value.

        Returns
        -------
        tuple[float, float]
            ``(l_occ, l_free)`` log-odds update values.
            Caller uses ``l_occ`` for hit cells, ``l_free`` for free cells.
        """
        decay = math.exp(-self.distance_decay * distance) * math.exp(
            -self.angle_decay * angle_of_incidence
        )
        l_occ = self.base_l_occ * decay
        l_free = self.base_l_free * decay
        return l_occ, l_free

    def compute_mass(
        self,
        distance: float,
        angle_of_incidence: float,
        cell_type: str = "occupied",
    ) -> tuple[float, float, float]:
        """Compute DS/TBM mass function consistent with log-odds.

        Uses the matching strategy specified at construction time
        (``self.matching``):

        - ``"betp"`` (default): BetP(O) = sigmoid(l).

          Derivation::

              p = sigmoid(l)
              m_O  = max(0, 2p - 1)
              m_F  = max(0, 1 - 2p)
              m_OF = 1 - |2p - 1|

          Verification: BetP(O) = m_O + m_OF/2 = p (exact).

        - ``"ppl"``: PPl(O) = sigmoid(l).

          Derivation (p >= 0.5): m_F = 0, m_OF = (1-p)/p, m_O = (2p-1)/p.
          Derivation (p < 0.5):  m_O = 0, m_OF = p/(1-p), m_F = (1-2p)/(1-p).

          Verification: PPl(O) = Pl(O)/(Pl(O)+Pl(F)) = p (exact).

        Parameters
        ----------
        distance:
            Distance from robot to cell in metres.
        angle_of_incidence:
            Angle of incidence in radians [0, pi/2].
        cell_type:
            ``"occupied"`` or ``"free"`` -- selects which log-odds value
            (l_occ or l_free) is used to derive the mass function.

        Returns
        -------
        tuple[float, float, float]
            ``(m_O, m_F, m_OF)`` mass values summing to 1.0.
        """
        l_occ, l_free = self.compute_log_odds(distance, angle_of_incidence)
        l = l_occ if cell_type == "occupied" else l_free
        p = 1.0 / (1.0 + math.exp(-l))

        if self.matching == "ppl":
            return _ppl_masses(p)
        else:  # "betp"
            return _betp_masses(p)
