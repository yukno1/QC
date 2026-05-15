from absl.testing import absltest
from numpy.ma.core import identity

from src.lib import ops


class OpsTest(absltest.TestCase):
    def test_id(self):
        identity = ops.Identity()
        self.assertEqual(identity[0, 0], 1)
        self.assertEqual(identity[0, 1], 0)
        self.assertEqual(identity[1, 0], 0)
        self.assertEqual(identity[1, 1], 1)

    def test_rk(self):
        self.assertTrue(ops.Rk(0).is_close(ops.Identity()))
        self.assertTrue(ops.Rk(1).is_close(ops.PauliZ()))
        self.assertTrue(ops.Rk(2).is_close(ops.Sgate()))
        self.assertTrue(ops.Rk(3).is_close(ops.Tgate()))

    def test_gates_root(self):
        t = ops.Tgate()
        self.assertTrue(t(t).is_close(ops.Phase()))
        v = ops.Vgate()
        self.assertTrue(v(v).is_close(ops.PauliX()))
        yr = ops.Yroot()
        self.assertTrue(yr(yr).is_close(ops.PauliY()))


if __name__ == "__main__":
    absltest.main()
