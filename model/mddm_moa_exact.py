import math


class _MDDMBase:
    def __init__(self, n=100, delta=1e-6):
        self.n = int(n)
        self.delta = float(delta)
        self.win = [] # sliding window f recent correctness bits
        self.pointer = 0 # how many items are in the window
        self.u_max = 0.0 # best weighted accuracy seen so far
        self.is_change_detected = False
        self.is_initialized = False

    def _reset_common(self):
        self.win = [0] * self.n
        self.pointer = 0
        self.u_max = 0.0
        self.is_change_detected = False

    def _push_bit(self, correct_bit: int):
        bit = int(bool(correct_bit))
        if self.pointer < self.n:
            self.win[self.pointer] = bit
            self.pointer += 1
        else:
            for i in range(self.n - 1):
                self.win[i] = self.win[i + 1]
            self.win[self.n - 1] = bit

    def update(self, correct_bit: int) -> bool:
        if self.is_change_detected or not self.is_initialized:
            self.reset()
            self.is_initialized = True

        self._push_bit(correct_bit)

        drift = False
        if self.pointer == self.n:
            u = self._u_weighted()
            self.u_max = u if self.u_max < u else self.u_max
            drift = (self.u_max - u) > self.eps #best past performance - current performance > threshold

        self.is_change_detected = drift
        return drift


class MDDM_G_Exact(_MDDMBase):
    """
    MOA-style MDDM_G geometric scheme.
    ratio, ratio^2, ratio^3, ...
    """

    def __init__(self, n=100, ratio=1.01, delta=1e-6):
        self.ratio = float(ratio)
        super().__init__(n=n, delta=delta)
        self.reset()

    def reset(self):
        self._reset_common()
        self.eps = math.sqrt(0.5 * self._cal_sigma() * math.log(1.0 / self.delta))

    def _cal_sigma(self):
        total = 0.0
        bound_sum = 0.0
        r = self.ratio
        for _ in range(self.n):
            total += r
            r *= self.ratio
        r = self.ratio
        for _ in range(self.n):
            bound_sum += (r / total) ** 2
            r *= self.ratio
        return bound_sum

    def _u_weighted(self):
        total_sum = 0.0
        win_sum = 0.0
        r = self.ratio
        for i in range(self.n):
            total_sum += r
            win_sum += self.win[i] * r
            r *= self.ratio
        return win_sum / total_sum


class MDDM_A_Exact(_MDDMBase):
    """
    MOA-style MDDM_A arithmetic scheme.
    1, 1+d, 1+2d, 1+3d, ...
    """

    def __init__(self, n=100, difference=0.01, delta=1e-6):
        self.difference = float(difference)
        super().__init__(n=n, delta=delta)
        self.reset()

    def reset(self):
        self._reset_common()
        self.eps = math.sqrt(0.5 * self._cal_sigma() * math.log(1.0 / self.delta))

    def _cal_sigma(self):
        total = 0.0
        sigma = 0.0
        for i in range(self.n):
            total += 1.0 + i * self.difference
        for i in range(self.n):
            sigma += ((1.0 + i * self.difference) / total) ** 2
        return sigma

    def _u_weighted(self):
        total_sum = 0.0
        win_sum = 0.0
        for i in range(self.n):
            weight = 1.0 + i * self.difference
            total_sum += weight
            win_sum += self.win[i] * weight
        return win_sum / total_sum


class MDDM_E_Exact(_MDDMBase):
    """
    MOA-style MDDM_E Euler scheme.
    1, e^λ, e^(2λ), e^(3λ), ...
    """

    def __init__(self, n=100, lambd=0.01, delta=1e-6):
        self.lambd = float(lambd)
        super().__init__(n=n, delta=delta)
        self.reset()

    def reset(self):
        self._reset_common()
        self.eps = math.sqrt(0.5 * self._cal_sigma() * math.log(1.0 / self.delta))

    def _cal_sigma(self):
        total = 0.0
        bound_sum = 0.0
        r = 1.0
        ratio = math.exp(self.lambd)
        for _ in range(self.n):
            total += r
            r *= ratio
        r = 1.0
        for _ in range(self.n):
            bound_sum += (r / total) ** 2
            r *= ratio
        return bound_sum

    def _u_weighted(self):
        total_sum = 0.0
        win_sum = 0.0
        r = 1.0
        ratio = math.exp(self.lambd)
        for i in range(self.n):
            total_sum += r
            win_sum += self.win[i] * r
            r *= ratio
        return win_sum / total_sum
