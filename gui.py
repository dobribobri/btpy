#  -*- coding: utf-8 -*-
from tkinter import *
from tkinter import ttk
import os
import copy
import dill
import threading
import numpy as np
import attenuation
from integration import Integration
from core import Initialize
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib
matplotlib.use("TkAgg")


class Model:
    def __init__(self):
        self.data = None

        self.year, self.month, self.day, self.label = StringVar(), StringVar(), StringVar(), StringVar()
        self.oxygen_model, self.water_vapor_model = StringVar(), StringVar()
        self.integration_method = StringVar()
        self.h_start, self.h_stop = DoubleVar(value=0.), DoubleVar(value=15.)
        self.nu_start, self.nu_stop, self.nu_step = DoubleVar(value=18.0), DoubleVar(value=27.2), DoubleVar(value=0.1)
        self.theta = DoubleVar(value=51.)
        self.relic_background = IntVar(value=1)

        self.progress = IntVar(value=0)

    def load_data(self, path='radiosonde.gridded') -> None:
        with open(path, 'rb') as dump:
            self.data = dill.load(dump)

    @property
    def session_keys(self) -> np.ndarray:
        return np.asarray(list(self.data.keys()), dtype=int)

    def get_current_key(self) -> tuple:
        return tuple([int(_) for _ in list(map(lambda _: _.get(), [self.year, self.month, self.day, self.label]))])

    def check_key(self) -> bool:
        key = self.get_current_key()
        if key in self.data.keys():
            return True
        return False

    def get_current_data(self) -> tuple:
        return self.data[self.get_current_key()]

    def get_current_state(self) -> dict:
        d = dict()
        for attr_name in self.__dict__.keys():
            if attr_name not in ['data']:
                d[attr_name] = copy.deepcopy(self.__getattribute__(attr_name).get())
        d['T'], d['P'], d['rho_rel'], d['alt'] = map(copy.deepcopy, self.get_current_data())
        return d

    def save(self):
        with open(os.path.join('.tmp', 'settings'), 'wb') as dump:
            dill.dump(self.get_current_state(), dump)

    @classmethod
    def load(cls):
        model = Model()
        model.load_data()
        try:
            with open(os.path.join('.tmp', 'settings'), 'rb') as dump:
                d = dill.load(dump)
                for attr_name, val in d.items():
                    try:
                        model.__getattribute__(attr_name).set(val)
                    except AttributeError:
                        pass
        except FileNotFoundError:
            pass
        return model


def compute():
    m.save()

    button_compute.config(state=DISABLED)

    global window, canvas, figure, ax

    if plot_new.get():
        window = Toplevel(root)
        window.protocol("WM_DELETE_WINDOW", erase)
        window.title('Яркостная температура')
        window.withdraw()

        figure, ax = plt.subplots(figsize=(6, 4))

        canvas = FigureCanvasTkAgg(figure, master=window)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, window)
        toolbar.update()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        plot_new.set(value=False)
        button_erase.config(state=NORMAL)

    window2 = Toplevel(root)
    window2.title('Подождите...')
    window2.geometry('{:.0f}x{:.0f}'.format(400, 35))
    window2.resizable(width=False, height=False)
    window2.attributes("-topmost", True)
    s = ttk.Style()
    s.theme_use("default")
    s.configure("TProgressbar", thickness=33)
    progressbar = ttk.Progressbar(window2, orient="horizontal", variable=m.progress, length=100, style="TProgressbar")
    progressbar.pack(side=TOP, fill=BOTH, padx=1, pady=1)

    def progress_callback(*_):
        if m.progress.get() >= 100:
            stop_listen.set()
            m.progress.set(0)

            with open(os.path.join('.tmp', 'results'), 'rb') as dump:
                results = np.load(dump)

            window2.destroy()
            button_compute.config(state=NORMAL)

            ax.plot(results[:, 0], results[:, 1])
            ax.set_xlabel(r'Частота $\nu$, ГГц')
            ax.set_ylabel(r'Яркостная температура, К')
            plt.grid(ls=':')
            plt.tight_layout()
            canvas.draw()

            clear()

            button_erase.config(state=NORMAL)
            window.deiconify()

    m.progress.trace('w', progress_callback)

    core = Initialize(**m.get_current_state())
    n_workers = os.cpu_count()
    threading.Thread(target=core, args=(n_workers,)).start()

    if stop_listen.is_set():
        stop_listen.clear()
    threading.Thread(target=listen).start()


def listen():
    while True:
        if stop_listen.is_set():
            break
        try:
            with open(os.path.join('.tmp', 'progress'), 'rb') as file:
                m.progress.set(dill.load(file))
        except FileNotFoundError:
            pass
        except EOFError:
            pass


def clear():
    stop_listen.set()
    try:
        os.remove(os.path.join('.tmp', 'results'))
        os.remove(os.path.join('.tmp', 'progress'))
    except FileNotFoundError:
        pass


def erase(destroy: bool = False):
    global window, figure, ax

    if isinstance(ax, Axes):
        ax.clear()
    if isinstance(window, Toplevel):
        window.destroy()
    plt.close(figure)

    plot_new.set(value=True)
    button_erase.config(state=DISABLED)

    button_compute.config(state=NORMAL)

    clear()

    if destroy:
        root.destroy()
        root.quit()


def check_key_callback(*_) -> None:
    if m.check_key():
        status_label.config(text='OK')
        button_compute.config(state=NORMAL)
    else:
        status_label.config(text='X')
        button_compute.config(state=DISABLED)


if __name__ == '__main__':

    stop_listen = threading.Event()
    clear()

    root = Tk()
    root.title('GUI')
    root.geometry('{:.0f}x{:.0f}'.format(700, 425))
    root.resizable(width=False, height=False)

    m = Model.load()

    m.load_data('radiosonde.gridded')

    main_menu = Menu(root)
    root.config(menu=main_menu)
    menu = [Menu(main_menu, tearoff=0)]
    menu[0].add_command(label='        Создать базу...        ', state=DISABLED)
    menu[0].add_command(label='        Подключить базу...        ', state=DISABLED)
    main_menu.add_cascade(label='   Опции   ', menu=menu[0])

    y_level = 40
    Label(root, text='Год:').place(relx=.05, y=y_level * 1, anchor="w")
    years = [str(v).zfill(4) for v in np.unique(m.session_keys[:, 0])]
    year_select = ttk.Combobox(values=years, width=7, cursor='hand2', textvariable=m.year)
    year_select.place(relx=.11, y=y_level * 1, anchor="w")
    year_select.current(len(years) - 1)
    Label(root, text='Месяц:').place(relx=.25, y=y_level * 1, anchor="w")
    months = [str(v).zfill(2) for v in np.unique(m.session_keys[:, 1])]
    month_select = ttk.Combobox(values=months, width=7, cursor='hand2', textvariable=m.month)
    month_select.place(relx=.33, y=y_level * 1, anchor="w")
    month_select.current(0)
    Label(root, text='День:').place(relx=.48, y=y_level * 1, anchor="w")
    days = [str(v).zfill(2) for v in np.unique(m.session_keys[:, 2])]
    day_select = ttk.Combobox(values=days, width=7, cursor='hand2', textvariable=m.day)
    day_select.place(relx=.55, y=y_level * 1, anchor="w")
    day_select.current(0)
    Label(root, text='Метка:').place(relx=.70, y=y_level * 1, anchor="w")
    labels = [str(v).zfill(2) for v in np.unique(m.session_keys[:, -1])]
    label_select = ttk.Combobox(values=labels, width=7, cursor='hand2', textvariable=m.label)
    label_select.place(relx=.78, y=y_level * 1, anchor="w")
    label_select.current(0)

    status_label = Label(root)
    status_label.place(relx=.92, y=y_level * 1, anchor="w")

    m.year.trace('w', check_key_callback)
    m.month.trace('w', check_key_callback)
    m.day.trace('w', check_key_callback)
    m.label.trace('w', check_key_callback)

    Label(root, text='Модель поглощения в кислороде:').place(relx=.05, y=y_level * 2, anchor="w")
    oxygen_models = [option.value for option in attenuation.Oxygen.Models]
    oxygen_model_select = ttk.Combobox(values=oxygen_models, width=42, cursor='hand2',
                                       textvariable=m.oxygen_model)
    oxygen_model_select.place(relx=.455, y=y_level * 2, anchor="w")
    oxygen_model_select.current(0)

    Label(root, text='Модель поглощения в водяном паре:').place(relx=.05, y=y_level * 3, anchor="w")
    water_vapor_models = [option.value for option in attenuation.WaterVapor.Models]
    water_vapor_model_select = ttk.Combobox(values=water_vapor_models, width=42, cursor='hand2',
                                            textvariable=m.water_vapor_model)
    water_vapor_model_select.place(relx=.455, y=y_level * 3, anchor="w")
    water_vapor_model_select.current(2)

    Label(root, text='Метод интегрирования:').place(relx=.05, y=y_level * 4, anchor="w")
    integration_methods = [option.value for option in Integration.Methods]
    integration_method_select = ttk.Combobox(values=integration_methods, width=50, cursor='hand2',
                                             textvariable=m.integration_method)
    integration_method_select.place(relx=.364, y=y_level * 4, anchor="w")
    integration_method_select.current(2)

    Label(root, text='Начальная высота (км):').place(relx=.05, y=y_level * 5, anchor="w")
    h_start_sb = Spinbox(root, from_=0., to=100., textvariable=m.h_start, width=10, format="%.3f", increment=0.001)
    h_start_sb.place(relx=.32, y=y_level * 5, anchor="w")
    Label(root, text='Конечная высота (км):').place(relx=.55, y=y_level * 5, anchor="w")
    h_start_sb = Spinbox(root, from_=0., to=100., textvariable=m.h_stop, width=10, format="%.3f", increment=0.001)
    h_start_sb.place(relx=.82, y=y_level * 5, anchor="w")

    Label(root, text='Начальная частота (ГГц):').place(relx=.05, y=y_level * 6, anchor="w")
    nu_start_sb = Spinbox(root, from_=1., to=350., textvariable=m.nu_start, width=10, format="%.1f", increment=0.2)
    nu_start_sb.place(relx=.32, y=y_level * 6, anchor="w")
    Label(root, text='Конечная частота (ГГц):').place(relx=.55, y=y_level * 6, anchor="w")
    nu_start_sb = Spinbox(root, from_=1., to=350., textvariable=m.nu_stop, width=10, format="%.1f", increment=0.2)
    nu_start_sb.place(relx=.82, y=y_level * 6, anchor="w")
    Label(root, text='Шаг по частоте (ГГц):').place(relx=.05, y=y_level * 7, anchor="w")
    nu_start_sb = Spinbox(root, from_=0.01, to=10., textvariable=m.nu_step, width=10, format="%.2f", increment=0.01)
    nu_start_sb.place(relx=.32, y=y_level * 7, anchor="w")

    relic_background_cb = Checkbutton(root, text='  Учитывать космический фон',
                                      variable=m.relic_background, onvalue=1, offvalue=0)
    relic_background_cb.place(relx=.54, y=y_level * 7, anchor="w")

    Label(root, text='Угол наблюдения от зенита (градусы):').place(relx=.05, y=y_level * 8, anchor="w")
    theta_sb = Spinbox(root, from_=0, to=60., textvariable=m.theta, width=10, format="%.2f", increment=1)
    theta_sb.place(relx=.46, y=y_level * 8, anchor="w")

    window, canvas, figure, ax = [None] * 4
    plot_new = BooleanVar(value=True)
    button_compute = Button(root, text="Вычислить", width=20, height=1, cursor='hand2')
    button_compute.place(relx=.35, y=y_level * 9.5, anchor="center")
    button_compute.config(command=compute)

    button_erase = Button(root, text="Сброс", width=20, height=1, cursor='hand2')
    button_erase.place(relx=.65, y=y_level * 9.5, anchor="center")
    button_erase.config(command=erase, state=DISABLED)

    check_key_callback()

    root.protocol("WM_DELETE_WINDOW", lambda: erase(destroy=True))

    root.mainloop()
