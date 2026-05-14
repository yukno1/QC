from typing import List, Iterable

import itertools


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
