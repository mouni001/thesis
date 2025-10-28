# mddm_moa_exact.py
import math
from collections import deque

class MDDM_G_Exact:
    """
    MOA-style MDDM_G:
      - window holds 1 for CORRECT, 0 for ERROR
      - weighted mean accuracy (newer points weigh slightly more)
      - keep best-so-far u_max; trigger when (u_max - u) > eps
    """
    def __init__(self, n=100, ratio=1.01, delta=1e-6):
        self.n = int(n); self.ratio = float(ratio); self.delta = float(delta)
        self.win = deque(maxlen=self.n)
        self._build_weights(); self._compute_epsilon()
        self.u_max = 0.0

    def _build_weights(self):
        ws = []
        v = self.ratio
        for _ in range(self.n):
            ws.append(v); v *= self.ratio
        s = sum(ws)
        self.w = [w/s for w in ws]               # normalized weights
        self.sum_sq = sum(w*w for w in self.w)   # ∑ w_i^2

    def _compute_epsilon(self):
        # eps = sqrt( 0.5 * (∑w_i^2) * ln(1/delta) )
        self.eps = math.sqrt(0.5 * self.sum_sq * math.log(1.0/self.delta))

    def _u_weighted(self):
        if len(self.win) < self.n: return None
        return sum(self.w[i] * self.win[i] for i in range(self.n))

    def reset(self):
        self.win.clear(); self.u_max = 0.0

    def update(self, correct_bit: int) -> bool:
        """correct_bit: 1 if prediction is correct, else 0."""
        self.win.append( int(bool(correct_bit)) )
        if len(self.win) < self.n:
            return False
        u = self._u_weighted()
        self.u_max = max(self.u_max, u)
        drift = (self.u_max - u) > self.eps
        if drift:
            self.reset()
        return drift
    
# mddm_moa_exact.py
import math
from collections import deque

class MDDM_A_Exact:
    """
    MDDM-A (Arithmetic weights)
      - window holds 1 for CORRECT, 0 for ERROR
      - weights grow linearly with recency: w_i ∝ (i+1), i=0 oldest
      - drift if (u_max - u) > eps, then reset
    """
    def __init__(self, n=100, delta=1e-6):
        self.n = int(n)
        self.delta = float(delta)
        self.win = deque(maxlen=self.n)
        self._build_weights()
        self._compute_epsilon()
        self.u_max = 0.0

    def _build_weights(self):
        # oldest index 0 -> weight 1; newest index n-1 -> weight n
        ws = [i + 1 for i in range(self.n)]
        s = sum(ws)
        self.w = [w / s for w in ws]               # normalized weights (v_i)
        self.sum_sq = sum(w * w for w in self.w)   # ∑ v_i^2

    def _compute_epsilon(self):
        # eps = sqrt( 0.5 * (∑ v_i^2) * ln(1/δ) )
        self.eps = math.sqrt(0.5 * self.sum_sq * math.log(1.0 / self.delta))

    def _u_weighted(self):
        if len(self.win) < self.n:
            return None
        # deque order: win[0] oldest ... win[n-1] newest
        return sum(self.w[i] * self.win[i] for i in range(self.n))

    def reset(self):
        self.win.clear()
        self.u_max = 0.0

    def update(self, correct_bit: int) -> bool:
        """correct_bit: 1 if prediction is correct, else 0."""
        self.win.append(int(bool(correct_bit)))
        if len(self.win) < self.n:
            return False
        u = self._u_weighted()
        self.u_max = max(self.u_max, u)
        drift = (self.u_max - u) > self.eps
        if drift:
            self.reset()
        return drift


class MDDM_E_Exact:
    """
    MDDM-E (Exponential/Euler weights)
      - weights grow exponentially with recency: w_i ∝ exp(beta * i)
      - beta > 0 emphasizes newest points more strongly than MDDM-G with small ratios
    """
    def __init__(self, n=100, beta=0.02, delta=1e-6):
        self.n = int(n)
        self.beta = float(beta)
        self.delta = float(delta)
        self.win = deque(maxlen=self.n)
        self._build_weights()
        self._compute_epsilon()
        self.u_max = 0.0

    def _build_weights(self):
        ws = [math.exp(self.beta * i) for i in range(self.n)]
        s = sum(ws)
        self.w = [w / s for w in ws]               # normalized weights (v_i)
        self.sum_sq = sum(w * w for w in self.w)   # ∑ v_i^2

    def _compute_epsilon(self):
        self.eps = math.sqrt(0.5 * self.sum_sq * math.log(1.0 / self.delta))

    def _u_weighted(self):
        if len(self.win) < self.n:
            return None
        return sum(self.w[i] * self.win[i] for i in range(self.n))

    def reset(self):
        self.win.clear()
        self.u_max = 0.0

    def update(self, correct_bit: int) -> bool:
        self.win.append(int(bool(correct_bit)))
        if len(self.win) < self.n:
            return False
        u = self._u_weighted()
        self.u_max = max(self.u_max, u)
        drift = (self.u_max - u) > self.eps
        if drift:
            self.reset()
        return drift
    
