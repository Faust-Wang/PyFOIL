import numpy as np
from scipy.optimize import fsolve
from typing import NoReturn, Union


class BP3333:

    def __init__(self, params: dict) -> None:

        """BP3333 Parameterisation.

        Parameters
        ----------
        params : dict
            Parameters to define aerofoil geometry.
        """

        self.params = params
        self.violates_constraints = False

        self.b1 = self._calculate_b1()
        self.b9 = self._calculate_b9()

    def cp_lt(self) -> NoReturn:
        raise NotImplementedError('BP3333::cp_lt()')

    def cp_tt(self) -> NoReturn:
        raise NotImplementedError('BP3333::cp_tt()')

    def cp_lc(self) -> NoReturn:
        raise NotImplementedError('BP3333::cp_lc()')

    def cp_tc(self) -> NoReturn:
        raise NotImplementedError('BP3333::cp_tc()')

    def generate(self) -> NoReturn:
        raise NotImplementedError('BP3333::generate()')

    def _calculate_b1(self) -> Union[float, None]:

        """Calculates the value of b1.

        Returns
        -------
        b1 : float or None
            Value of b1 calculated, None if constraints violated.
        """

        s1 = (16 + 3 * self.params['k_c'] * (
                np.tan(self.params['gamma_le']) ** -1 +
                np.tan(self.params['alpha_te']) ** -1
        ) * (
                1 + self.params['z_te'] *
                np.tan(self.params['alpha_te']) ** -1
        ))

        s2 = (6 * self.params['k_c'] * (
                np.tan(self.params['gamma_le']) ** -1 +
                np.tan(self.params['alpha_te']) ** -1
        ) * (1 - self.params['y_c'] * (
                np.tan(self.params['gamma_le']) ** -1 +
                np.tan(self.params['alpha_te']) ** -1
        ) + (self.params['z_te'] * np.tan(self.params['alpha_te'])) ** -1))

        s2 = np.array([-1, 1]) * 4 * (16 + s2) ** 0.5

        b1 = ((s1 + s2) / (
                3 * self.params['k_c'] * (
                np.tan(self.params['gamma_le']) ** -1 +
                np.tan(self.params['alpha_te']) ** -1
        ) ** 2))

        b1 = list(filter(lambda x: self._b1constraint(x), b1))

        if len(b1) == 0:
            self.violates_constraints = True
            return None
        else:
            return min(b1)

    def _calculate_b9(self) -> Union[float, None]:

        """Calculates the value of b9.

        Returns
        -------
        b9 : float or None
            Value of b9 calculated, None if constraints violated.
        """

        def _b9equation(b9, x_t, y_t, r_le, k_t):

            b9expr = (
                    27 * k_t ** 2 * b9 ** 4 / 4 -
                    27 * k_t ** 2 * x_t * b9 ** 3 +
                    (9 * k_t * y_t + 81 * k_t ** 2 * x_t ** 2 / 2) * b9 ** 2 +
                    (2 * r_le - 18 * k_t * x_t * y_t -
                     27 * k_t ** 2 * x_t ** 3) * b9 +
                    3 * y_t ** 2 +
                    9 * k_t * x_t ** 2 * y_t +
                    27 * k_t ** 2 * x_t ** 4 / 4
            )

            return b9expr

        b9 = fsolve(
            func=_b9equation,
            x0=0.15,
            args=(self.params['x_t'],
                  self.params['y_t'],
                  self.params['r_le'],
                  self.params['k_t'])
        )

        b9 = list(filter(lambda x: self._b9constraint(x), b9))

        if len(b9) == 0:
            self.violates_constraints = True
            return None
        else:
            return min(b9)

    def _b1constraint(self, b: float):

        """Checks if b1 satisfies constraints.

        Parameters
        ----------
        b : float
            Value of b1 to check.

        Returns
        -------
        bool
            True if satisfies constraints, False otherwise.
        """

        c1 = 0 < b < self.params['y_c']

        c2 = self.params['x_c'] - (
                2 * (b - self.params['y_c']) / (3 * self.params['k_c'])
        ) ** 0.5 > b / np.tan(self.params['gamma_le'])

        c3 = (1 + (self.params['z_te'] - b) * np.tan(self.params['alpha_te']) ** -1) > (
                self.params['x_c'] +
                (2 * (b - self.params['y_c']) / (3 * self.params['k_c'])) ** 0.5
        )

        c4 = np.isreal(b)

        return c1 and c2 and c3 and c4

    def _b9constraint(self, b: float) -> bool:

        """Checks if b9 satisfies constraints.

        Parameters
        ----------
        b : float
            Value of b9 to check.

        Returns
        -------
        bool
            True if satisfies constraints, False otherwise.
        """

        c1 = 0 < b < self.params['x_t']
        c2 = (self.params['x_t'] -
              (-2 * self.params['y_t'] / (3 * self.params['k_t'])) ** 0.5) < b
        c3 = (1 + (self.params['dz_te'] -
                   (3 * self.params['k_t'] * (self.params['x_t'] - b) ** 2 / 2 +
                    self.params['y_t'])) /
              np.tan(self.params['beta_te'])) > 2 * self.params['x_t'] - b

        return c1 and c2 and c3
