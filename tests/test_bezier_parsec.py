import numpy as np
import pytest
from pyfoil.bezier_parsec import BP3333


class TestBP3333:

    @pytest.fixture
    def bp3333(self):

        params = {
            'r_le': -0.037808837823291544,
            'x_t': 0.3330169248396024,
            'y_t': 0.0860606161531148,
            'k_t': -0.02123278455954114,
            'dz_te': 0.00016016656552749232,
            'beta_te': 0.253672169082407,
            'gamma_le': 0.07217486810664123,
            'x_c': 0.4778046347032303,
            'y_c': 0.011077115412001716,
            'k_c': -0.07634236236745753,
            'z_te': 0.008782006669331481,
            'alpha_te': 0.052736424828297046
        }

        bp = BP3333(params)

        return bp

    def test_cp_lt(self, bp3333):

        arr_cps = np.asarray(bp3333.cp_lt())
        target_cps = np.array([
            [0, 0],
            [0, 0.0860108750248033],
            [0.2934976733657358, 0.0860606161531148],
            [0.3330169248396024, 0.0860606161531148]
        ])

        assert np.array_equal(arr_cps, target_cps)

    def test_cp_tt(self, bp3333):

        arr_cps = np.asarray(bp3333.cp_tt())
        target_cps = np.array([
            [0.3330169248396024, 0.0860606161531148],
            [0.37253617631346897, 0.0860606161531148],
            [0.6688589262976907, 0.0860108750248033],
            [1, 0.00016016656552749232]
        ])

        assert np.array_equal(arr_cps, target_cps)

    def test_cp_lc(self, bp3333):

        arr_cps = np.asarray(bp3333.cp_lc())
        target_cps = np.array([
            [0, 0],
            [0.03907176271172611, 0.0028249062027728747],
            [0.20935878875041608, 0.011077115412001716],
            [0.4778046347032303, 0.011077115412001716]
        ])

        assert np.array_equal(arr_cps, target_cps)

    def test_cp_tc(self, bp3333):

        arr_cps = np.asarray(bp3333.cp_tc())
        target_cps = np.array([
            [0.4778046347032303, 0.011077115412001716],
            [0.7462504806560446, 0.011077115412001716],
            [1.112855146522983, 0.0028249062027728747],
            [1, 0.008782006669331481]
        ])

        assert np.array_equal(arr_cps, target_cps)

    def test_generate(self, bp3333):

        x_u, y_u, x_l, y_l = bp3333.generate()

        assert isinstance(x_u, np.ndarray)
        assert isinstance(y_u, np.ndarray)
        assert isinstance(x_l, np.ndarray)
        assert isinstance(y_l, np.ndarray)

    def test__calculate_b1(self, bp3333):

        assert bp3333.b1 == 0.0028249062027728747

    def test__calculate_b9(self, bp3333):

        assert bp3333.b9 == 0.2934976733657358
