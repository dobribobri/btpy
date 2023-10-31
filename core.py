#  -*- coding: utf-8 -*-
import os
import dill
import numpy as np
import attenuation
from integration import Integration, at
from vapor import absolute_humidity
from multiprocessing import Pool
from tqdm import tqdm


class Tqdm(tqdm):
    def __init__(self, *args, **kwargs):
        self.k = 0
        super().__init__(*args, **kwargs)

    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self, value):
        self.__n = value
        percent = int(value / self.total * 100.)
        if percent >= 100:
            percent = 99
        if self.k % 10 == 0:
            with open(os.path.join('.tmp', 'progress'), 'wb') as file:
                dill.dump(percent, file)
        self.k += 1


class Initialize:
    def __init__(self, **kwargs):
        self.oxygen_model, self.water_vapor_model = [''] * 2
        self.integration_method = ''
        self.h_start, self.h_stop = [0.] * 2
        self.nu_start, self.nu_stop, self.nu_step = [0.] * 3
        self.theta = 0.
        self.relic_background = True
        self.T, self.P, self.rho_rel, self.alt = [np.array([])] * 4
        self.rho = np.array([])

        for name, val in kwargs.items():
            self.__setattr__(name, val)

        cond = (self.h_start <= self.alt) & (self.alt <= self.h_stop)
        self.T, self.P, self.rho_rel, self.alt = map(lambda _: _[cond], [self.T, self.P, self.rho_rel, self.alt])
        self.rho = absolute_humidity(self.T, self.P, self.rho_rel)
        self.dh = np.diff(np.insert(self.alt, 0, self.h_start))
        self.sec = 1. / np.cos(self.theta * np.pi / 180.)
        self.frequencies = np.arange(self.nu_start, self.nu_stop + self.nu_step, self.nu_step)

    def bt_downwelling(self, nu: float):

        g = self.sec * (attenuation.Oxygen.gamma(model=self.oxygen_model,
                                                 frequency=nu,
                                                 T=self.T, P=self.P, rho=self.rho) +
                        attenuation.WaterVapor.gamma(model=self.water_vapor_model,
                                                     frequency=nu,
                                                     T=self.T, P=self.P, rho=self.rho))
        T = self.T + 273.15

        def f(h):
            integral = Integration.integrate(method=self.integration_method, a=g, lower=0, upper=h, dh=self.dh)
            return at(T, h) * at(g, h) * np.exp(-1 * integral)

        inf = len(g) - 1
        brt = Integration.integrate_callable(method=self.integration_method, f=f, lower=0, upper=inf, dh=self.dh)

        background = 0.
        if self.relic_background:
            tau = Integration.integrate(method=self.integration_method, a=g, lower=0, upper=inf, dh=self.dh)
            background = 2.72548 * np.exp(-1 * tau)

        return nu, brt + background

    def __call__(self, n_workers: int = 1):
        if not os.path.exists('.tmp'):
            os.makedirs('.tmp')

        results = []

        with Pool(processes=n_workers) as pool:
            for result in Tqdm(pool.imap_unordered(self.bt_downwelling, self.frequencies),
                               total=len(self.frequencies)):
                results.append(result)

        results = np.asarray(sorted(results, key=lambda _: _[0]))

        with open(os.path.join('.tmp', 'results'), 'wb') as dump:
            np.save(dump, results)

        with open(os.path.join('.tmp', 'progress'), 'wb') as file:
            dill.dump(100, file)
