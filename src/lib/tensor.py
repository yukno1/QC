import numpy as np
from absl import flags
from typing import TypeVar
import math

T = TypeVar("T", bound="Tensor")

flags.DEFINE_integer("tensor_width", 64, "Width of complex (64, 128")


def tensor_width():
    """Return global floating point bit width"""
    try:
        return flags.FLAGS.tensor_width
    except:
        return 64


def tensor_type():
    """Return complex type based on command-line flag."""
    assert tensor_width() == 64 or tensor_width() == 128
    return np.complex64 if tensor_width() == 64 else np.complex128


class Tensor(np.ndarray):

    def __new__(cls, input_array, op_name=None) -> T:
        cls.name = op_name
        return np.asarray(input_array, dtype=tensor_type()).view(cls)

    def __array_finalize__(self, obj) -> None:
        if obj is None:
            return

    @property
    def nbits(self) -> int:
        return int(math.log2(self.shape[0]))

    def kron(self, arg: Tensor) -> Tensor:
        return self.__class__(np.kron(self, arg))

    def __mul__(self, arg: Tensor) -> T:
        return self.kron(arg)

    def kpow(self, n: int) -> T:
        if n == 0:
            return self.__class__(1.0)
        t = self
        for _ in range(n - 1):
            t = np.kron(t, self)
        return self.__class__(t)  # return a tensor type

    def is_close(self, arg) -> bool:
        return np.allclose(self, arg, atol=1e-6)

    def is_hermitian(self) -> bool:
        if len(self.shape) != 2 or self.shape[0] != self.shape[1]:
            return False
        return self.is_close(np.conj(self.transpose()))

    def is_unitary(self) -> bool:
        return Tensor(np.conj(self.transpose()) @ self).is_close(
            Tensor(np.eye(self.shape[0]))
        )

    def is_permutation(self) -> bool:
        x = self
        return (
            x.ndim == 2
            and x.shape[0] == x.shape[1]
            and (x.sum(axis=0) == 1).all()
            and (x.sum(axis=1) == 1).all()
            and ((x == 1) or (x == 0)).all()
        )
