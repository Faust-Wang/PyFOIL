import numpy as np
from pyfoil.utils.bezier import bernstein, curve, dcurve


class TestBezier:

    def test_bernstein(self):

        fn = bernstein(3, 1)

        assert fn(0.5) == 0.375
        assert fn(1.0) == 0.0

    def test_curve(self):

        odcp = [1, 3, 5, 7]
        tdcp = [[0, 0], [1, 2], [2, 5], [3, 9]]

        obc = curve(odcp, num=5)
        tbc = curve(tdcp, num=5)

        obc_target = np.array([
            [1.0],
            [2.5],
            [4.0],
            [5.5],
            [7.0]
        ])

        tbc_target = np.array([
            [0.0, 0.0],
            [0.75, 1.6875],
            [1.5, 3.75],
            [2.25, 6.1875],
            [3.0, 9.0]
        ])

        assert obc.shape[0] == tbc.shape[0] == 5
        assert obc.shape[1] == 1
        assert tbc.shape[1] == 2

        assert np.array_equal(obc, obc_target)
        assert np.array_equal(tbc, tbc_target)

    def test_dcurve(self):

        odcp = [1, 3, 5, 7]
        tdcp = [[0, 0], [1, 2], [2, 5], [3, 9]]

        odbc = dcurve(odcp, num=5)
        tdbc = dcurve(tdcp, num=5)

        odbc_target = np.array([
            [6.0],
            [6.0],
            [6.0],
            [6.0],
            [6.0]
        ])

        tdbc_target = np.array([
            [3.0, 6.0],
            [3.0, 7.5],
            [3.0, 9.0],
            [3.0, 10.5],
            [3.0, 12.0]
        ])

        assert odbc.shape[0] == tdbc.shape[0] == 5
        assert odbc.shape[1] == 1
        assert tdbc.shape[1] == 2

        assert np.array_equal(odbc, odbc_target)
        assert np.array_equal(tdbc, tdbc_target)
