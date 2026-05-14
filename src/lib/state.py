import math
import cmath
import random

import numpy as np
from typing import Optional, List
import helper
import tensor


class State(tensor.Tensor):
    """class State represents single- and multi-qubit state"""

    def ampl(self, *bits) -> np.complexfloating:
        return self[helper.bits2val(bits)]

    def prob(self, *bits) -> float:
        amplitude = self.ampl(*bits)
        return np.real(amplitude.conj() * amplitude).item()

    def maxprob(self) -> tuple[List[float], float]:
        idx = np.abs(self).argmax()
        maxprob = np.real(self[idx].conj() * self[idx])
        maxbits = helper.val2bits(idx, self.nbits)
        return maxbits, maxprob

    def normalize(self):
        dprod = np.conj(self) @ self
        assert not dprod.is_close(0.0), "Normalizing to 0-probability state"
        self /= np.sqrt(np.real(dprod))  # modify in-place
        return self

    def phase(self, *bits) -> float:
        amplitude = self.ampl(*bits)
        return math.degrees(cmath.phase(amplitude))

    def dump(self, desc: str = None, prob_only: bool = True) -> None:
        pass

    def density(self) -> tensor.Tensor:
        return tensor.Tensor(np.outer(self, self.conj()))


def qubit(alpha: Optional[complex] = None, beta: Optional[complex] = None) -> State:
    if alpha is None and beta is None:
        raise ValueError("alpha, beta or both, need to be specified")
    if beta is None:
        beta = np.sqrt(1.0 - np.real(np.conj(alpha) * alpha)).item()
    if alpha is None:
        alpha = np.sqrt(1.0 - np.real(np.conj(beta) * beta)).item()

    norm = np.real(np.conj(alpha) * alpha) + np.real(np.conj(beta) * beta)
    assert math.isclose(norm, 1.0), "Qubit probabilities not equal to 1."
    return State([alpha, beta])


def zeros_or_ones(d: int = 1, idx: int = 0) -> State:
    assert d > 0, "Need to specify at least 1 qubit"
    t = np.zeros(2**d, dtype=tensor.tensor_type())
    t[idx] = 1.0
    return State(t)


def zeros(d: int = 1) -> State:
    return zeros_or_ones(d, 0)


def ones(d: int = 1) -> State:
    return zeros_or_ones(d, 2**d - 1)


def bitstring(*bits) -> State:
    arr = np.asarray(bits)
    assert len(arr) > 0, "Need to specify at least 1 qubit"
    assert ((arr == 1) | (arr == 0)).all(), "Bits must be 0 or 1"

    t = np.zeros(1 << len(bits), dtype=tensor.tensor_type())
    t[helper.bits2val(bits)] = 1
    return State(t)


def rand_bits(n: int) -> State:
    bits = [random.randint(0, 1) for _ in range(n)]
    return bitstring(*bits)
