import cmath
import math
import numpy as np
from typing import Union, List

from src.lib import tensor
from src.lib import state


class Operator(tensor.Tensor):
    """Operators are represented as square, unitary matrices."""

    def __new__(subtype, input_array, name=None) -> Operator:
        obj = super().__new__(subtype, input_array)
        obj.name = name
        return obj

    def adjoint(self) -> Operator:
        return self.__class__(np.conj(self.transpose()))

    def dump(self, desc=None, digits=3) -> None:
        np.set_printoptions(precision=digits)
        if desc:
            print(f"{desc} ({self.nbits}-qubit(s) operator)")
        print(self)

    def apply(
        self, arg: Union[state.State, Operator], idx: int
    ) -> Union[state.State, Operator]:
        if isinstance(arg, Operator):
            arg_bits = arg.nbits
            if idx > 0:
                arg = Operator(Identity().kpow(idx) * arg)
            if self.nbits > arg.nbits:
                arg = Operator(arg * Identity().kpow(self.nbits - idx - arg_bits))
            assert self.nbits == arg.nbits, "Mismatched dimensions."
            return self.__class__(arg @ self)

        assert isinstance(arg, state.State), "Error, expected State."
        op = self
        if idx > 0:
            op = Identity().kpow(idx) * op
        if arg.nbits - idx - self.nbits > 0:
            op = op * Identity().kpow(arg.nbits - idx - self.nbits)
        # note the reversed parameters.
        return state.State(np.matmul(self, arg))

    def __call__(
        self, arg: Union[state.State, Operator], idx: int = 0
    ) -> Union[state.State, Operator]:
        return self.apply(arg, idx)


# --------------------------------
# Single-Qubit Gates / Generators.
# --------------------------------


def Identity(d: int = 1) -> Operator:
    return Operator([[1.0, 0.0], [0.0, 1.0]], "Id").kpow(d)


def PauliX(d: int = 1) -> Operator:
    return Operator([[0.0, 1.0], [1.0, 0.0]], "X").kpow(d)


def PauliY(d: int = 1) -> Operator:
    return Operator([[0.0, -1.0j], [1.0j, 0.0]], "Y").kpow(d)


def PauliZ(d: int = 1) -> Operator:
    return Operator([[1.0, 0.0], [0.0, -1.0]], "Z").kpow(d)


def Rotation(vparm: List[float], theta: float, name: str = None) -> Operator:
    """Produce the single-qubit rotation operator."""
    v = np.asarray(vparm)
    ...
    return Operator(
        np.cos(theta / 2) * Identity()
        - 1j
        * np.sin(theta / 2)
        * (v[0] * PauliX() + v[1] * PauliY() + v[2] * PauliZ()),
        name,
    )


def RotationX(theta: float) -> Operator:
    return Rotation([1.0, 0.0, 0.0], theta, "Rx")


def RotationY(theta: float) -> Operator:
    return Rotation([0.0, 1.0, 0.0], theta, "Ry")


def RotationZ(theta: float) -> Operator:
    return Rotation([0.0, 0.0, 1.0], theta, "Rz")


def Hadamard(d: int = 1) -> Operator:
    return Operator(1 / np.sqrt(2) * np.array([[1.0, 1.0], [1.0, -1.0]]), "H").kpow(d)


def Phase(d: int = 1) -> Operator:
    return Operator([[1.0, 0.0], [0.0, 1.0j]], "S").kpow(d)


Sgate = Phase


# T-gate is sqrt(S
def Tgate(d: int = 1) -> Operator:
    return Operator([[1.0, 0.0], [0.0, cmath.exp(1j * math.pi / 4)]], "T").kpow(d)


def Vgate(d: int = 1) -> Operator:
    return Operator(0.5 * np.array([(1 + 1j, 1 - 1j), (1 - 1j, 1 + 1j)]), "V").kpow(d)


def Yroot(d: int = 1) -> Operator:
    return Operator(
        0.5 * np.array([(1 + 1j, -1 - 1j), (1 + 1j, 1 + 1j)]), "Yroot"
    ).kpow(d)


# IBM's U1-gate
def U1(lam: float, d: int = 1) -> Operator:
    return Operator([[1.0, 0.0], [0.0, cmath.exp(1j * lam)]], "U1").kpow(d)


# IBM's general U3-gate
def U3(theta: float, phi: float, lam: float, d: int = 1) -> Operator:
    return Operator(
        [
            [np.cos(theta / 2), -cmath.exp(1j * lam) * np.sin(theta / 2)],
            [
                cmath.exp(1j * phi) * np.sin(theta / 2),
                cmath.exp(1j * (phi + lam)) * np.cos(theta / 2),
            ],
        ],
        "U3",
    ).kpow(d)


def Rk(k: int, d: int = 1) -> Operator:
    return U1(2 * math.pi / (2**k)).kpow(d)


def ZeroProject(nbits: int) -> Operator:
    zero_projector = np.zeros((2**nbits, 2**nbits))
    zero_projector[0, 0] = 1
    return Operator(zero_projector)


def OneProjector(nbits: int) -> Operator:
    dim = 2**nbits
    zero_projector = np.zeros((dim, dim))
    zero_projector[dim - 1, dim - 1] = 1
    return Operator(zero_projector)
