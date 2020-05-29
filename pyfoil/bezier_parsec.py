from typing import List, Tuple, Union

import numpy as np
from scipy.optimize import fsolve
from .utils.bezier import curve, dcurve


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

    def cp_lt(self) -> List[List[float]]:

        x0 = 0
        x1 = 0
        x2 = self.b9
        x3 = self.params['x_t']

        y0 = 0
        y1 = (3 * self.params['k_t'] *
              (self.params['x_t'] - self.b9) ** 2 / 2 + self.params['y_t'])
        y2 = self.params['y_t']
        y3 = self.params['y_t']

        cps = [
            [x0, y0],
            [x1, y1],
            [x2, y2],
            [x3, y3]
        ]

        return cps

    def cp_tt(self) -> List[List[float]]:

        x0 = self.params['x_t']
        x1 = 2 * self.params['x_t'] - self.b9
        x2 = 1 + (self.params['dz_te'] - (
                3 * self.params['k_t'] *
                (self.params['x_t'] - self.b9) ** 2 / 2 + self.params['y_t']
        )) / np.tan(self.params['beta_te'])
        x3 = 1

        y0 = self.params['y_t']
        y1 = self.params['y_t']
        y2 = (3 * self.params['k_t'] *
              (self.params['x_t'] - self.b9) ** 2 / 2 + self.params['y_t'])
        y3 = self.params['dz_te']

        cps = [
            [x0, y0],
            [x1, y1],
            [x2, y2],
            [x3, y3]
        ]

        return cps

    def cp_lc(self) -> List[List[float]]:

        x0 = 0
        x1 = self.b1 * np.tan(self.params['gamma_le']) ** -1
        x2 = self.params['x_c'] - (
                2 * (self.b1 - self.params['y_c']) / (3 * self.params['k_c'])
        ) ** 0.5
        x3 = self.params['x_c']

        y0 = 0
        y1 = self.b1
        y2 = self.params['y_c']
        y3 = self.params['y_c']

        cps = [
            [x0, y0],
            [x1, y1],
            [x2, y2],
            [x3, y3]
        ]

        return cps

    def cp_tc(self) -> List[List[float]]:

        x0 = self.params['x_c']
        x1 = self.params['x_c'] + (
                2 * (self.b1 - self.params['y_c']) / (3 * self.params['k_c'])
        ) ** 0.5
        x2 = 1 + (self.params['z_te'] - self.b1) * np.tan(
            self.params['alpha_te']
        ) ** -1
        x3 = 1

        y0 = self.params['y_c']
        y1 = self.params['y_c']
        y2 = self.b1
        y3 = self.params['z_te']

        cps = [
            [x0, y0],
            [x1, y1],
            [x2, y2],
            [x3, y3]
        ]

        return cps

    def generate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        """Generates Aerofoil Profile.

        Returns
        -------
        x_u : np.ndarray
            x-coordinates of the upper surface.
        y_u : np.ndarray
            y-coordinates of the upper surface.
        x_l : np.ndarray
            x-coordinates of the lower surface.
        y_l : np.ndarray
            y-coordinates of the lower surface.
        """

        cp_lt = self.cp_lt()
        cp_tt = self.cp_tt()
        cp_lc = self.cp_lc()
        cp_tc = self.cp_tc()

        thickness = np.concatenate((curve(cp_lt), curve(cp_tt)))
        camber = np.concatenate((curve(cp_lc), curve(cp_tc)))
        yt = np.interp(camber[:, 0], thickness[:, 0], thickness[:, 1])

        dc = np.concatenate((dcurve(cp_lc), dcurve(cp_tc)))
        theta = np.arctan(dc[:, 1] / dc[:, 0])

        x_u = camber[:, 0] - yt / 2 * np.sin(theta)
        x_l = camber[:, 0] + yt / 2 * np.sin(theta)
        y_u = camber[:, 1] + yt / 2 * np.cos(theta)
        y_l = camber[:, 1] - yt / 2 * np.cos(theta)

        return x_u, y_u, x_l, y_l

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
        ) + self.params['z_te'] * np.tan(self.params['alpha_te']) ** -1))

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

        c3 = (1 + (self.params['z_te'] - b) * np.tan(
            self.params['alpha_te']) ** -1) > (
                     self.params['x_c'] +
                     (2 * (b - self.params['y_c']) / (
                             3 * self.params['k_c'])) ** 0.5
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
