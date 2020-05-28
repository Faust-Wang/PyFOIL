import numpy as np
from scipy.special import binom
from typing import Callable, Sequence, Union


def bernstein(n: int, k: int) -> Callable[[np.ndarray], np.ndarray]:

    """Generates expression for the Bernstein polynomial.

    Parameters
    ----------
    n : int
        Total number of elements in the set.
    k : int
        Number of elements to choose from the set.

    Returns
    -------
    bernstein_polynomial : function
        Functional expression for the Bernstein Polynomial.
    """

    binomial_coefficient = binom(n, k)

    def bernstein_polynomial(x: np.ndarray) -> np.ndarray:
        return binomial_coefficient * x ** k * (1 - x) ** (n - k)

    return bernstein_polynomial


def curve(points: Sequence[Union[float, Sequence[float]]], num: int = 50) -> np.ndarray:

    """Generates a Bezier Curve from a collection of points.

    The function takes a collection of Bezier control points and
    calculates the corresponding Bezier curve - this works for any
    dimensionality as long as the input array is 1D / 2D.

    For example, to generate a Bezier Curve for 3D control points:

      points = [
          [0, 0, 0],
          [3, 5, 7],
          [6, 8, 9]
      ]

    This will return an (n, 3) numpy array of coordinates.

    Parameters
    ----------
    points : list
        Control points used to generate the Bezier Curve.
    num : int
        Number of points used to generate the Bezier Curve.

    Returns
    -------
    bezier_curve : np.ndarray
        Coordinates of the generated Bezier Curve.
    """

    arr_points = np.asarray(points)

    if not 0 < arr_points.ndim <= 2:
        raise ValueError('points must be 1D / 2D')

    n = len(points)
    t = np.linspace(0, 1, num)
    bezier_curve = np.zeros((num, arr_points.ndim))

    for i in range(n):
        bezier_curve += np.outer(bernstein(n - 1, i)(t), arr_points[i])

    return bezier_curve


def dcurve(points: Sequence[Union[float, Sequence[float]]], num: int = 50) -> np.ndarray:

    """Derivatives of a Bezier Curve from a collection of points.

    The function takes a collection of Bezier control points and
    calculates the derivatives of the corresponding Bezier curve - this
    works for any dimensionality as long as the input array is 1D / 2D.

    For example, to calculate the derivatives of a 3D Bezier curve:

      points = [
          [0, 0, 0],
          [3, 5, 7],
          [6, 8, 9]
      ]

    This will return an (n, 3) numpy array of coordinates.

    Parameters
    ----------
    points : list
        Control points used to generate the Bezier Curve.
    num : int
        Number of points used to generate the Bezier Curve.

    Returns
    -------
    bezier_derivative : np.ndarray
        Derivatives of the generated Bezier Curve.
    """

    arr_points = np.asarray(points)

    if not 0 < arr_points.ndim <= 2:
        raise ValueError('points must be 1D / 2D')

    n = len(points)
    t = np.linspace(0, 1, num)
    bezier_derivative = np.zeros((num, arr_points.ndim))

    for i in range(n - 1):
        bezier_derivative += np.outer(
            bernstein(n - 2, i)(t), arr_points[i + 1] - arr_points[i]
        )

    bezier_derivative *= (n - 1)

    return bezier_derivative
