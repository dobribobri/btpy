#  -*- coding: utf-8 -*-
from typing import Union
import numpy as np

"""
Водяной пар
"""


def pressure(T: Union[float, np.ndarray], rho: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Парциальное давление водяного пара

    :param rho: плотность водяного пара (абсолютная влажность), г/м^3
    :param T: температура воздуха, град. Цельс.
    :return: давление в гПа
    """
    return rho * (T + 273.15) / 216.7


def relative_humidity(T: Union[float, np.ndarray], P: Union[float, np.ndarray],
                      rho: Union[float, np.ndarray], method='wmo2008') -> Union[float, np.ndarray]:
    """
    Расчет относительной влажности по абсолютной

    :param T: температура воздуха, град. Цельс.
    :param P: барометрическое давление, гПа
    :param rho: абсолютная влажность, г/м^3
    :param method: метод расчета давления насыщенного водяного пара
        ('wmo2008', 'august-roche-magnus', 'tetens', 'august', 'buck')
    :return: %
    """
    return pressure(T, rho) / saturated.pressure(T, P, method) * 100


def absolute_humidity(T: Union[float, np.ndarray], P: Union[float, np.ndarray],
                      rel: Union[float, np.ndarray], method='wmo2008') -> Union[float, np.ndarray]:
    """
    Расчет абсолютной влажности по относительной

    :param T: температура воздуха, град. Цельс.
    :param P: барометрическое давление, гПа
    :param rel: относительная влажность, %
    :param method: метод расчета давления насыщенного водяного пара
        ('wmo2008', 'august-roche-magnus', 'tetens', 'august', 'buck')
    :return: г/м^3
    """
    return (rel / 100) * 216.7 * saturated.pressure(T, P, method) / (T + 273.15)


class saturated:
    """
    Насыщенный водяной пар
    """

    @staticmethod
    def pressure(T: Union[float, np.ndarray], P: Union[float, np.ndarray] = None,
                 method='wmo2008') -> Union[float, np.ndarray]:
        """
        Давление насыщенного водяного пара во влажном воздухе

        :param T: температура воздуха, град. Цельс.
        :param P: барометрическое давление, гПа
        :param method: метод аппроксимации ('wmo2008', 'august-roche-magnus',
            'tetens', 'august', 'buck')
        :return: давление в гПа
        """
        if method.lower() == 'august-roche-magnus':
            e = 0.61094 * np.exp(17.625 * T / (243.04 + T)) * 10
        elif method.lower() == 'tetens':
            e = 0.61078 * np.exp(17.27 * T / (T + 237.3)) * 10
        elif method.lower() == 'august':
            e = np.exp(20.386 - 5132 / (T + 273.15)) * 1.333
        elif method.lower() == 'buck':
            if T > 0:
                e = 6.1121 * np.exp((18.678 - T / 234.5) * (T / (257.14 + T)))
            else:
                e = 6.1115 * np.exp((23.036 - T / 333.7) * (T / (279.82 + T)))
        else:
            e = 6.112 * np.exp(17.62 * T / (243.12 + T))
        if P is None:
            return e
        return (1.0016 + 3.15 * 0.000001 * P - 0.074 / P) * e
