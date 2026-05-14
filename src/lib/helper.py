from typing import List, Iterable, Tuple
import itertools
import numpy as np


def bits2val(bits: List[int]) -> int:
    return sum(v * (1 << (len(bits) - i - 1)) for i, v in enumerate(bits))


def val2bits(val: int, nbits: int):
    return [int(c) for c in format(val, "0{}b".format(nbits))]


def bitprod(nbits: int) -> Iterable[int]:
    for bits in itertools.product([0, 1], repeat=nbits):
        yield bits


def bit2frac(bits: Iterable) -> float:
    return sum(bit * 2 ** (-idx - 1) for idx, bit in enumerate(bits))


def frac2bits(val: float, nbits: int):
    assert val < 1.0, "frac2bits: value must be strictly < 1.0"
    res = []
    while nbits:
        nbits -= 1
        val *= 2
        res.append(int(val))
        val -= int(val)
    return res


def density_to_cartesian(rho: np.ndarray) -> Tuple[float, float, float]:
    a = rho[0, 0]
    c = rho[1, 0]
    x = 2.0 * c.real
    y = 2.0 * c.imag
    z = 2.0 * a - 1.0
    return np.real(x), np.real(y), np.real(z)
