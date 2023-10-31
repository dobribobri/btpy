#  -*- coding: utf-8 -*-
from typing import Union, Callable
from enum import Enum
import numpy as np


def diap(a: Union[float, np.ndarray], start: int, stop: int, step: int = 1) -> Union[float, np.ndarray]:
    rank = np.ndim(a)
    if rank == 0:
        return a
    if rank == 1:
        return a[start:stop:step]
    if rank == 3:
        return a[:, :, start:stop:step]
    raise RuntimeError('wrong rank')


def at(a: Union[float, np.ndarray], index: int) -> Union[float, np.ndarray]:
    rank = np.ndim(a)
    if rank == 0:
        return a
    if rank == 1:
        return a[index]
    if rank == 3:
        return a[:, :, index]
    raise RuntimeError('wrong rank')


class Integration:
    class Methods(Enum):
        TRAPZ = '1. Метод трапеций'
        SIMPSON = '2. Формула Симпсона'
        BOOLE = '3. Правило Буля'

    @staticmethod
    def trapz(a: np.ndarray, lower: int, upper: int, dh: np.ndarray) -> np.ndarray:
        return np.sum(diap(a, lower + 1, upper) * diap(dh, lower + 1, upper), axis=-1) + \
            (at(a, lower) * at(dh, lower) + at(a, upper) * at(dh, upper)) / 2.

    @staticmethod
    def simpson(a: np.ndarray, lower: int, upper: int, dh: np.ndarray) -> np.ndarray:
        return (at(a, lower) * at(dh, lower) + at(a, upper) * at(dh, upper) +
                4 * np.sum(diap(a, lower + 1, upper, 2) * diap(dh, lower + 1, upper, 2), axis=-1) +
                2 * np.sum(diap(a, lower + 2, upper, 2) * diap(dh, lower + 2, upper, 2), axis=-1)) / 3.

    @staticmethod
    def boole(a: np.ndarray, lower: int, upper: int, dh: np.ndarray) -> np.ndarray:
        return (14 * (at(a, lower) * at(dh, lower) + at(a, upper) * at(dh, upper)) +
                64 * np.sum(diap(a, lower + 1, upper, 2) * diap(dh, lower + 1, upper, 2), axis=-1) +
                24 * np.sum(diap(a, lower + 2, upper, 4) * diap(dh, lower + 2, upper, 4), axis=-1) +
                28 * np.sum(diap(a, lower + 4, upper, 4) * diap(dh, lower + 4, upper, 4), axis=-1)) / 45.

    @staticmethod
    def integrate(method: str, a: np.ndarray, lower: int, upper: int, dh: np.ndarray) -> np.ndarray:
        if method == Integration.Methods.TRAPZ.value:
            return Integration.trapz(a, lower, upper, dh)
        if method == Integration.Methods.SIMPSON.value:
            return Integration.simpson(a, lower, upper, dh)
        # default
        return Integration.boole(a, lower, upper, dh)

    @staticmethod
    def integrate_callable(method: str, f: Callable, lower: int, upper: int, dh: np.ndarray) -> np.ndarray:
        a = np.asarray([f(i) for i in range(lower, upper + 1, 1)])
        if np.ndim(a) == 3:
            a = np.transpose(a, axes=(1, 2, 0))
        return Integration.integrate(method, a, lower, upper, dh)
