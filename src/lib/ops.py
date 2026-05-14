import tensor
import numpy as np
import state
from typing import Union


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
