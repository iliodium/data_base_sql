import os
import glob
import uuid
import random
import psycopg2
import numpy as np
import scipy.interpolate
import scipy.io as sio
import matplotlib.cm as cm
import matplotlib.tri as mtri
import matplotlib.pyplot as plt

from docx import Document
from docx.shared import Inches, Pt, Mm
from docx.enum.text import WD_ALIGN_PARAGRAPH

from psycopg2 import Error
from celluloid import Camera
from scipy.fft import fft, rfftfreq


class Artist:
    """Класс отвечает за отрисовку графиков

    Виды графиков:
        -Изополя:
            -Максимальные значения
            -Средние значения
            -Минимальные значения
            -Среднее квадратичное отклонение

        -Огибающие:
            -Максимальные значения
            -Минимальные значения

        -Коэффициенты обеспеченности:
            -Максимальные значения
            -Минимальные значения

        -Спектры

        -Нестационарные сигналы

    """

    @staticmethod
    def interpolator(coords, val):
        return scipy.interpolate.RBFInterpolator(coords, val, kernel='cubic')

    @staticmethod
    def film(coeffs, coordinates, model_name):
        print('START')
        min_v = np.min(coeffs)
        max_v = np.max(coeffs)
        #  Шаг для изополей и контурных линий
        levels = np.arange(np.round(min_v, 1), np.round(max_v, 1) + 0.1, 0.1)
        x, z = np.array(coordinates)

        breadth, depth, height = int(model_name[0]) / 10, int(model_name[1]) / 10, int(model_name[2]) / 10
        count_sensors_on_model = len(coeffs[0])
        count_sensors_on_middle = int(model_name[0]) * 5
        count_sensors_on_side = int(model_name[1]) * 5
        count_row = count_sensors_on_model // (2 * (count_sensors_on_middle + count_sensors_on_side))
        x = np.reshape(x, (count_row, -1))
        x = np.split(x, [count_sensors_on_middle,
                         count_sensors_on_middle + count_sensors_on_side,
                         2 * count_sensors_on_middle + count_sensors_on_side,
                         2 * (count_sensors_on_middle + count_sensors_on_side)
                         ], axis=1)
        z = np.reshape(z, (count_row, -1))
        z = np.split(z, [count_sensors_on_middle,
                         count_sensors_on_middle + count_sensors_on_side,
                         2 * count_sensors_on_middle + count_sensors_on_side,
                         2 * (count_sensors_on_middle + count_sensors_on_side)
                         ], axis=1)

        del x[4]
        del z[4]

        for i in range(4):
            x[i] = x[i].tolist()
            z[i] = z[i].tolist()

        x1, x2, x3, x4 = x  # x это тензор со всеми координатами граней по ширине,x1...x4 координаты отдельных граней
        z1, z2, z3, z4 = z  # z это тензор со всеми координатами граней по высоте,z1...z4 координаты отдельных граней

        x = [np.array(x1), np.array(x2), np.array(x3), np.array(x4)]  # Входные координаты для изополей
        z = [np.array(z1), np.array(z2), np.array(z3), np.array(z4)]  # Входные координаты для изополей
        ret_int = []  # массив с функциями изополей

        # Расширение матрицы координат по бокам
        for i in range(len(x1)):
            x1[i] = [0] + x1[i] + [breadth]
            x2[i] = [breadth] + x2[i] + [breadth + depth]
            x3[i] = [breadth + depth] + x3[i] + [2 * breadth + depth]
            x4[i] = [2 * breadth + depth] + x4[i] + [2 * (breadth + depth)]

        x1.append(x1[0])
        x2.append(x2[0])
        x3.append(x3[0])
        x4.append(x4[0])

        x1.insert(0, x1[0])
        x2.insert(0, x2[0])
        x3.insert(0, x3[0])
        x4.insert(0, x4[0])

        x_extended = [np.array(x1), np.array(x2), np.array(x3), np.array(x4)]  # Расширенные координаты для изополей

        # Расширение матрицы координат по бокам
        for i in range(len(z1)):
            z1[i] = [z1[i][0]] + z1[i] + [z1[i][0]]
            z2[i] = [z2[i][0]] + z2[i] + [z2[i][0]]
            z3[i] = [z3[i][0]] + z3[i] + [z3[i][0]]
            z4[i] = [z4[i][0]] + z4[i] + [z4[i][0]]

        z1.append([0 for _ in range(len(z1[0]))])
        z2.append([0 for _ in range(len(z2[0]))])
        z3.append([0 for _ in range(len(z3[0]))])
        z4.append([0 for _ in range(len(z4[0]))])

        z1.insert(0, [height for _ in range(len(z1[0]))])
        z2.insert(0, [height for _ in range(len(z2[0]))])
        z3.insert(0, [height for _ in range(len(z3[0]))])
        z4.insert(0, [height for _ in range(len(z4[0]))])

        z_extended = [np.array(z1), np.array(z2), np.array(z3), np.array(z4)]  # Расширенные координаты для изополей

        fig, graph = plt.subplots(1, 4, figsize=(16, 9))
        camera = Camera(fig)

        cmap = cm.get_cmap(name="jet")

        x_new = []
        x_old = []
        z_new = []
        z_old = []
        coords = []
        data_new = []
        for i in range(4):
            # x это координаты по ширине
            x_new.append(x_extended[i].reshape(1, -1)[0])
            x_old.append(x[i].reshape(1, -1)[0])
            # Вычитаем чтобы все координаты по x находились в интервале [0, 1]
            if i == 1:
                x_old[i] = x_old[i] - breadth
                x_new[i] = x_new[i] - breadth
            elif i == 2:
                x_old[i] = x_old[i] - (breadth + depth)
                x_new[i] = x_new[i] - (breadth + depth)
            elif i == 3:
                x_old[i] = x_old[i] - (2 * breadth + depth)
                x_new[i] = x_new[i] - (2 * breadth + depth)
            # z это координаты по высоте
            z_new.append(z_extended[i].reshape(1, -1)[0])
            z_old.append(z[i].reshape(1, -1)[0])
            coords.append([[i1, j1] for i1, j1 in zip(x_old[i], z_old[i])])

            graph[i].set_ylim([0, height])
            if breadth == depth == height:
                graph[i].set_aspect('equal')
            if i in [0, 2]:
                graph[i].set_xlim([0, breadth])
                graph[i].set_xticks(ticks=np.arange(0, breadth + 0.1, 0.1))
            else:
                graph[i].set_xlim([0, depth])
                graph[i].set_xticks(ticks=np.arange(0, depth + 0.1, 0.1))

        qqq = 0
        for pressure_coefficients in coeffs[:300]:
            qqq += 1
            print(qqq)

            pressure_coefficients = np.reshape(pressure_coefficients, (count_row, -1))
            pressure_coefficients = np.split(pressure_coefficients,
                                             [count_sensors_on_middle,
                                              count_sensors_on_middle + count_sensors_on_side,
                                              2 * count_sensors_on_middle + count_sensors_on_side,
                                              2 * (
                                                      count_sensors_on_middle + count_sensors_on_side)
                                              ], axis=1)
            del pressure_coefficients[4]

            for i in range(4):
                data_old = pressure_coefficients[i].reshape(1, -1)[0]
                # Интерполятор полученный на основе имеющихся данных
                interpolator = Artist.interpolator(coords[i], data_old)
                # Получаем данные для несуществующих датчиков
                data_new = [float(interpolator([[X, Y]])) for X, Y in zip(x_new[i], z_new[i])]
                triang = mtri.Triangulation(x_new[i], z_new[i])
                refiner = mtri.UniformTriRefiner(triang)
                grid, value = refiner.refine_field(data_new, subdiv=4)
                data_colorbar = graph[i].tricontourf(grid, value, cmap=cmap, levels=levels, extend='both')
                aq = graph[i].tricontour(grid, value, linewidths=1, linestyles='solid', colors='black', levels=levels)
                graph[i].clabel(aq, fontsize=13)
            camera.snap()

        cbar = fig.colorbar(data_colorbar, ax=graph, location='bottom', cmap=cmap, ticks=levels)
        cbar.ax.tick_params(labelsize=5)
        animation = camera.animate(interval=125)
        animation.save('wind.gif')

        plt.clf()
        plt.close()

    @staticmethod
    def isofield(mode, pressure_coefficients, coordinates, alpha, model_name, angle):
        """Отрисовка изополей"""
        integral_func = []
        mods = {
            'max': np.max(pressure_coefficients, axis=0),
            'mean': np.mean(pressure_coefficients, axis=0),
            'min': np.min(pressure_coefficients, axis=0),
            'std': np.std(pressure_coefficients, axis=0),
        }  # Виды изополей
        pressure_coefficients = mods[mode]
        min_v = np.min(pressure_coefficients)
        max_v = np.max(pressure_coefficients)
        steps = {
            'max': 0.2,
            'mean': 0.2 if alpha == 6 else 0.1,
            'min': 0.2,
            'std': 0.05,
        }  # Шаги для изополей
        #  Шаг для изополей и контурных линий
        levels = np.arange(np.round(min_v, 1), np.round(max_v, 1) + 0.1, steps[mode])
        x, z = np.array(coordinates)
        # if integral[1] == 1:
        #     # Масштабирование
        #     print(model_name, 1.25)
        #     k = 1.25  # коэффициент масштабирования по высоте
        #     z *= k
        # else:
        #     k = 1
        #     print(model_name, 1)
        breadth, depth, height = int(model_name[0]) / 10, int(model_name[1]) / 10, int(model_name[2]) / 10
        count_sensors_on_model = len(pressure_coefficients)
        count_sensors_on_middle = int(model_name[0]) * 5
        count_sensors_on_side = int(model_name[1]) * 5
        count_row = count_sensors_on_model // (2 * (count_sensors_on_middle + count_sensors_on_side))
        pressure_coefficients = np.reshape(pressure_coefficients, (count_row, -1))
        pressure_coefficients = np.split(pressure_coefficients, [count_sensors_on_middle,
                                                                 count_sensors_on_middle + count_sensors_on_side,
                                                                 2 * count_sensors_on_middle + count_sensors_on_side,
                                                                 2 * (count_sensors_on_middle + count_sensors_on_side)
                                                                 ], axis=1)
        x = np.reshape(x, (count_row, -1))
        x = np.split(x, [count_sensors_on_middle,
                         count_sensors_on_middle + count_sensors_on_side,
                         2 * count_sensors_on_middle + count_sensors_on_side,
                         2 * (count_sensors_on_middle + count_sensors_on_side)
                         ], axis=1)

        z = np.reshape(z, (count_row, -1))
        z = np.split(z, [count_sensors_on_middle,
                         count_sensors_on_middle + count_sensors_on_side,
                         2 * count_sensors_on_middle + count_sensors_on_side,
                         2 * (count_sensors_on_middle + count_sensors_on_side)
                         ], axis=1)

        del pressure_coefficients[4]
        del x[4]
        del z[4]

        for i in range(4):
            x[i] = x[i].tolist()
            z[i] = z[i].tolist()

        x1, x2, x3, x4 = x  # x это тензор со всеми координатами граней по ширине,x1...x4 координаты отдельных граней
        z1, z2, z3, z4 = z  # z это тензор со всеми координатами граней по высоте,z1...z4 координаты отдельных граней

        x = [np.array(x1), np.array(x2), np.array(x3), np.array(x4)]  # Входные координаты для изополей
        z = [np.array(z1), np.array(z2), np.array(z3), np.array(z4)]  # Входные координаты для изополей
        ret_int = []  # массив с функциями изополей
        # Расширение матрицы координат по бокам
        for i in range(len(x1)):
            x1[i] = [0] + x1[i] + [breadth]
            x2[i] = [breadth] + x2[i] + [breadth + depth]
            x3[i] = [breadth + depth] + x3[i] + [2 * breadth + depth]
            x4[i] = [2 * breadth + depth] + x4[i] + [2 * (breadth + depth)]

        x1.append(x1[0])
        x2.append(x2[0])
        x3.append(x3[0])
        x4.append(x4[0])

        x1.insert(0, x1[0])
        x2.insert(0, x2[0])
        x3.insert(0, x3[0])
        x4.insert(0, x4[0])

        x_extended = [np.array(x1), np.array(x2), np.array(x3), np.array(x4)]  # Расширенные координаты для изополей

        # Расширение матрицы координат по бокам
        for i in range(len(z1)):
            z1[i] = [z1[i][0]] + z1[i] + [z1[i][0]]
            z2[i] = [z2[i][0]] + z2[i] + [z2[i][0]]
            z3[i] = [z3[i][0]] + z3[i] + [z3[i][0]]
            z4[i] = [z4[i][0]] + z4[i] + [z4[i][0]]

        z1.append([0 for _ in range(len(z1[0]))])
        z2.append([0 for _ in range(len(z2[0]))])
        z3.append([0 for _ in range(len(z3[0]))])
        z4.append([0 for _ in range(len(z4[0]))])

        z1.insert(0, [height for _ in range(len(z1[0]))])
        z2.insert(0, [height for _ in range(len(z2[0]))])
        z3.insert(0, [height for _ in range(len(z3[0]))])
        z4.insert(0, [height for _ in range(len(z4[0]))])

        z_extended = [np.array(z1), np.array(z2), np.array(z3), np.array(z4)]  # Расширенные координаты для изополей
        fig, graph = plt.subplots(1, 4)
        cmap = cm.get_cmap(name="jet")
        data_colorbar = None
        # data_old_integer = []  # данные для дискретного интегрирования по осям
        # data_for_3d_model = []  # данные для 3D модели
        for i in range(4):
            # x это координаты по ширине
            x_new = x_extended[i].reshape(1, -1)[0]
            x_old = x[i].reshape(1, -1)[0]
            # Вычитаем чтобы все координаты по x находились в интервале [0, 1]
            if i == 1:
                x_old -= breadth
                x_new -= breadth
            elif i == 2:
                x_old -= (breadth + depth)
                x_new -= (breadth + depth)
            elif i == 3:
                x_old -= (2 * breadth + depth)
                x_new -= (2 * breadth + depth)
            # z это координаты по высоте
            z_new = z_extended[i].reshape(1, -1)[0]
            z_old = z[i].reshape(1, -1)[0]
            data_old = pressure_coefficients[i].reshape(1, -1)[0]
            # data_old_integer.append(data_old)
            coords = [[i1, j1] for i1, j1 in zip(x_old, z_old)]  # Старые координаты
            # Интерполятор полученный на основе имеющихся данных
            interpolator = Artist.interpolator(coords, data_old)
            integral_func.append(interpolator)
            # Получаем данные для несуществующих датчиков
            data_new = [float(interpolator([[X, Y]])) for X, Y in zip(x_new, z_new)]

            triang = mtri.Triangulation(x_new, z_new)
            refiner = mtri.UniformTriRefiner(triang)
            grid, value = refiner.refine_field(data_new, subdiv=4)
            # data_for_3d_model.append((grid, value))
            data_colorbar = graph[i].tricontourf(grid, value, cmap=cmap, levels=levels, extend='both')
            aq = graph[i].tricontour(grid, value, linewidths=1, linestyles='solid', colors='black', levels=levels)

            graph[i].clabel(aq, fontsize=13)
            graph[i].set_ylim([0, height])
            if breadth == depth == height:
                graph[i].set_aspect('equal')
            if i in [0, 2]:
                graph[i].set_xlim([0, breadth])
                graph[i].set_xticks(ticks=np.arange(0, breadth + 0.1, 0.1))
            else:
                graph[i].set_xlim([0, depth])
                graph[i].set_xticks(ticks=np.arange(0, depth + 0.1, 0.1))
            ret_int.append(interpolator)
            # graph[i].axis('off')
        fig.colorbar(data_colorbar, ax=graph, location='bottom', cmap=cmap, ticks=levels)

        plt.savefig(f'{folder_out}\\isofield {model_name}_{alpha} {mode}')
        # plt.show()
        plt.clf()
        plt.close()
        # isofield_model().show(data_for_3d_model, levels)
        # return ret_int
        # # Численное интегрирование
        #
        # if integral[0] == -1:
        #     return None
        # elif integral[0] == 0:
        #     count_zone = count_row
        # else:
        #     count_zone = integral[0]
        # ran = random.uniform
        # face1, face2, face3, face4 = [], [], [], []
        # model = face1, face2, face3, face4
        # count_point = integral[1]
        # step_floor = height / count_row
        # for i in range(4):
        #     top = step_floor
        #     down = 0
        #     for _ in range(count_zone):
        #         floor = []
        #         if i in [0, 2]:
        #             for _ in range(count_point):
        #                 floor.append(integral_func[i]([[ran(0, breadth), ran(down, top)]]))
        #         else:
        #             for _ in range(count_point):
        #                 floor.append(integral_func[i]([[ran(0, depth), ran(down, top)]]))
        #         down = top
        #         top += step_floor
        #         # Обычный метод Монте-Синякина
        #         if i in [0, 2]:
        #             model[i].append(sum(floor) * breadth / count_point)
        #         else:
        #             model[i].append(sum(floor) * depth / count_point)
        # print(face1)
        # print(face2)
        # print(face3)
        # print(face4)

    @staticmethod
    def func(x):
        return 15 * x ** 3 + 21 * x ** 2 + 41 * x + 3 * np.sin(x) * np.cos(x)

    @staticmethod
    def check():
        arr = []
        a = 0
        b = 5
        n = 2000000
        for i in range(n):
            arr.append(Artist.func(random.uniform(a, b)))

        print(sum(arr) * (b - a) / n)

    @staticmethod
    def signal(pressure_coefficients, pressure_coefficients1, alpha, model_name, angle):
        time = [i / 1000 for i in range(5001)]
        plt.plot(time, [i[0] for i in pressure_coefficients[:5001]], label='113')
        plt.plot(time, [i[0] for i in pressure_coefficients1[:5001]], label='115')
        plt.legend()
        plt.show()

    @staticmethod
    def signal1(pressure_coefficients, pressure_coefficients_2):
        time = [i / 1000 for i in range(len(pressure_coefficients))]
        plt.plot(time, [i[206] for i in pressure_coefficients], label='old')
        plt.plot(time, pressure_coefficients_2, label='new')
        plt.legend()
        plt.show()

    @staticmethod
    def spectrum(pressure_coefficients, pressure_coefficients1, alpha, model_name, angle):
        N = len(pressure_coefficients)
        yf = (1 / N) * (np.abs(fft([i[0] for i in pressure_coefficients])))[1:N // 2]
        yf1 = (1 / N) * (np.abs(fft([i[0] for i in pressure_coefficients1])))[1:N // 2]
        FD = 1000
        xf = rfftfreq(N, 1 / FD)[1:N // 2]
        xf = xf[1:]
        yf = yf[1:]
        yf1 = yf1[1:]
        plt.plot(xf[:200], yf[:200], antialiased=True, label='113')
        plt.plot(xf[:200], yf1[:200], antialiased=True, label='115')
        plt.legend()
        plt.show()


class Controller:
    """
    Через класс происходит управление базой данных:
        -создание таблиц
        -заполнение таблиц
        -генерация несуществующих вариантов
    """
    __connection = None
    __path_database = None
    __path_database = 'D:\Projects\mat_to_csv\mat files'
    __extrapolatedAnglesInfoList = {}  # ключ вида T<model_name>_<alpha>_<angle>, значения (коэффициенты, x, z)

    def __init__(self):
        self.cursor = None

    def connect(self, database = '', password = '', user = "postgres", host = "127.0.0.1", port = "5432"):
        """Метод подключается от PostgreSQL"""
        try:
            self.__connection = psycopg2.connect(user=user,
                                                 password=password,
                                                 host=host,
                                                 port=port,
                                                 database=database)
            self.cursor = self.__connection.cursor()
            print("Соединение с PostgreSQL открыто")
        except psycopg2.OperationalError:
            assert False, ("Проверьте правильность веденных данных")
        except Exception as error:
            assert False, (f"Ошибка - {error}")

    @staticmethod
    def read_mat(path):
        """Считывает и обрабатывает данные из .mat файла"""
        alpha = path[-11:-10]
        mat_file = sio.loadmat(path)
        breadth = float(mat_file['Building_breadth'][0][0])
        depth = float(mat_file['Building_depth'][0][0])
        height = float(mat_file['Building_height'][0][0])
        model_name = int(f'{int(breadth * 10)}{int(depth * 10)}{int(height * 10)}')
        frequency = int(mat_file['Sample_frequency'][0][0])
        period = float(mat_file['Sample_period'][0][0])
        speed = float(mat_file['Uh_AverageWindSpeed'][0])
        x = [float('%.5f' % i) for i in mat_file["Location_of_measured_points"][0]]
        z = [float('%.5f' % i) for i in mat_file["Location_of_measured_points"][1]]
        sensor_number = [int(i) for i in mat_file["Location_of_measured_points"][2]]
        face_number = [int(i) for i in mat_file["Location_of_measured_points"][3]]
        angle = int(mat_file['Wind_direction_angle'][0][0])
        pressure_coefficients = mat_file["Wind_pressure_coefficients"] * 1000
        pressure_coefficients = pressure_coefficients.round(0)
        pressure_coefficients = pressure_coefficients.astype('int32')
        pressure_coefficients = pressure_coefficients.tolist()
        return {'alpha': alpha,
                'model_name': model_name,
                'breadth': breadth,
                'depth': depth,
                'height': height,
                'frequency': frequency,
                'period': period,
                'speed': speed,
                'x': x,
                'z': z,
                'sensor_number': sensor_number,
                'face_number': face_number,
                'angle': angle,
                'pressure_coefficients': pressure_coefficients,
                }

    @staticmethod
    def converter_coordinates(x_old, depth, breadth, face_number, count_sensors):
        """Возвращает из (x_old) -> (x,y)"""
        x = []
        y = []
        for i in range(count_sensors):
            if face_number[i] == 1:
                x.append(float('%.5f' % (-depth / 2)))
                y.append(float('%.5f' % (breadth / 2 - x_old[i])))
            elif face_number[i] == 2:
                x.append(float('%.5f' % (- depth / 2 + x_old[i] - breadth)))
                y.append(float('%.5f' % (-breadth / 2)))
            elif face_number[i] == 3:
                x.append(float('%.5f' % (depth / 2)))
                y.append(float('%.5f' % (-3 * breadth / 2 + x_old[i] - depth)))
            else:
                x.append(float('%.5f' % (3 * depth / 2 - x_old[i] + 2 * breadth)))
                y.append(float('%.5f' % (breadth / 2)))

        return x, y

    def fill_models_alpha(self, parameters):
        """Добавляет запись в таблицу models_alpha_<alpha>"""
        alpha = parameters['alpha']
        model_name = parameters['model_name']
        angle = parameters['angle']
        pressure_coefficients = parameters['pressure_coefficients']
        flag = 0
        try:
            name = f"T{model_name}_{alpha}_{angle:03d}_1.mat"
            if alpha == '6':
                self.cursor.execute("""
                           select model_id 
                           from experiments_alpha_6
                           where model_name = (%s)
                       """, (model_name,))
            elif alpha == '4':
                self.cursor.execute("""
                           select model_id 
                           from experiments_alpha_4
                           where model_name = (%s)
                       """, (model_name,))

            model_id = self.cursor.fetchall()[0][0]
            if alpha == '6':
                self.cursor.execute("""
                                       select model_id 
                                       from models_alpha_6
                                       where model_id = (%s) and angle = (%s)
                                        """, (model_id, angle))
            elif alpha == '4':
                self.cursor.execute("""
                           select model_id 
                           from models_alpha_4
                           where model_id = (%s) and angle = (%s)
                            """, (model_id, angle))
            check_exists = self.cursor.fetchall()
            if check_exists:
                print(f'{name} была ранее добавлена в models_alpha_{alpha}')
                flag = 1

            if flag == 0:
                try:
                    if alpha == '6':
                        self.cursor.execute("""
                                   insert into models_alpha_6
                                   values
                                   ((%s), (%s), (%s))
                               """, (model_id, angle, pressure_coefficients))
                    elif alpha == '4':
                        self.cursor.execute("""
                                   insert into models_alpha_4
                                   values
                                   ((%s), (%s), (%s))
                               """, (model_id, angle, pressure_coefficients))
                    self.__connection.commit()
                    print(f'{name} добавлена в models_alpha_{alpha}')
                except (Exception, Error) as error:
                    print(f"Ошибка при работе с PostgreSQL, файл {model_name}", error)
        except (Exception, Error) as error:
            print(f"Ошибка при работе с PostgreSQL, файл {model_name}", error)

    def fill_experiments_alpha(self, parameters):
        """Добавляет запись в таблицу experiments_alpha_<alpha>"""
        alpha = parameters['alpha']
        angle = parameters['angle']
        model_name = parameters['model_name']
        breadth = parameters['breadth']
        depth = parameters['depth']
        height = parameters['height']
        frequency = parameters['frequency']
        period = parameters['period']
        speed = parameters['speed']
        x = parameters['x']
        z = parameters['z']
        sensor_number = parameters['sensor_number']
        face_number = parameters['face_number']
        flag = 0  # флаг для проверки наличия записи в бд
        name = f"T{model_name}_{alpha}_{angle:03d}_1.mat"
        try:
            if alpha == '6':
                self.cursor.execute("""
                           select model_name 
                           from experiments_alpha_6
                       """)
            elif alpha == '4':
                self.cursor.execute("""
                           select model_name 
                           from experiments_alpha_4
                       """)
            table = self.cursor.fetchall()
            for row in table:
                if row[0] == model_name:
                    print(f'{name} была ранее добавлена в experiments_alpha_{alpha}')
                    flag = 1
                    break
            if flag == 0:
                try:
                    if alpha == '6':
                        self.cursor.execute("""
                                   insert into experiments_alpha_6 (model_name,
                                                                    breadth,
                                                                    depth,
                                                                    height,
                                                                    sample_frequency,
                                                                    sample_period,
                                                                    uh_AverageWindSpeed,
                                                                    x_coordinates,
                                                                    z_coordinates,
                                                                    sensor_number,
                                                                    face_number)
                                   values((%s),(%s),(%s),(%s),(%s),(%s),(%s),(%s),(%s),(%s),(%s))
                               """, (
                            model_name, breadth, depth, height, frequency, period, speed, x, z, sensor_number,
                            face_number))
                    elif alpha == '4':
                        self.cursor.execute(f"""
                                   insert into experiments_alpha_4 (model_name,
                                                                    breadth,
                                                                    depth,
                                                                    height,
                                                                    sample_frequency,
                                                                    sample_period,
                                                                    uh_AverageWindSpeed,
                                                                    x_coordinates,
                                                                    z_coordinates,
                                                                    sensor_number,
                                                                    face_number)
                                   values((%s),(%s),(%s),(%s),(%s),(%s),(%s),(%s),(%s),(%s),(%s))
                               """, (
                            model_name, breadth, depth, height, frequency, period, speed, x, z, sensor_number,
                            face_number))
                    self.__connection.commit()
                    print(f'{name} добавлена в experiments_alpha_{alpha}')
                except (Exception, Error) as error:
                    print(f"Ошибка при работе с PostgreSQL, файл {model_name}", error)
        except (Exception, Error) as error:
            print(f"Ошибка при работе с PostgreSQL, файл {model_name}", error)

    def fill_db(self, path = None):
        """
        Метод заполняет таблицу experiments_alpha_<alpha> и models_alpha_<alpha> данными из .mat файла.

        Если путь не передан, то в таблицы добавляются все данные из директории.
        """

        if path:
            parameters = self.read_mat(path)
            self.fill_experiments_alpha(parameters)
            self.fill_models_alpha(parameters)

        else:
            self.check_path()
            for alpha in os.listdir(self.__path_database):
                for files in os.listdir(f'{self.__path_database}\\{alpha}'):
                    for file in os.listdir(f"{self.__path_database}\\{alpha}\\{files}"):
                        self.fill_db(f"{self.__path_database}\\{alpha}\\{files}\\{file}")

    def get_paths(self):
        """Возвращает список путей всех файлов"""
        self.paths = []
        self.check_path()
        for alpha in os.listdir(self.__path_database):
            for files in os.listdir(f'{self.__path_database}\\{alpha}'):
                for file in os.listdir(f"{self.__path_database}\\{alpha}\\{files}"):
                    self.paths.append(f"{self.__path_database}\\{alpha}\\{files}\\{file}")

        return self.paths

    def check_path(self):
        """Метод проверяет наличие заданного пути"""
        if self.__path_database is None:
            self.__path_database = input('Введите полный путь до данных:\n')

    def disconnect(self):
        """Метод отключается от PostgreSQL"""
        if self.__connection:
            self.cursor.close()
            self.__connection.close()
            print("Соединение с PostgreSQL закрыто")

    def create_tables(self):
        """Метод создаёт таблицы"""
        # создается таблица с экспериментами
        try:
            self.cursor.execute('''
                create table if not exists experiments_alpha_4
                (
                    model_id serial primary key,
                    model_name smallint not null,
                    breadth real not null,
                    depth real not null,
                    height real not null,
                    sample_frequency smallint not null,
                    sample_period real not null,
                    uh_AverageWindSpeed real not null,
                    x_coordinates real[] not null,
                    z_coordinates real[] not null,
                    sensor_number smallint[] not null,
                    face_number smallint[] not null
                )''')
            self.__connection.commit()
            print("Создана таблица с экспериментами с параметром 4")
        except Exception as error:
            print(f"Ошибка -> {error}")
            self.__connection.commit()

        try:
            self.cursor.execute('''
                create table if not exists experiments_alpha_6
                (
                    model_id serial primary key,
                    model_name smallint not null,
                    breadth real not null,
                    depth real not null,
                    height real not null,
                    sample_frequency smallint not null,
                    sample_period real not null,
                    uh_AverageWindSpeed real not null,
                    x_coordinates real[] not null,
                    z_coordinates real[] not null,
                    sensor_number smallint[] not null,
                    face_number smallint[] not null
                )''')
            self.__connection.commit()
            print("Создана таблица с экспериментами с параметром 6")
        except Exception as error:
            print(f"Ошибка -> {error}")
            self.__connection.commit()

        # создается таблица с моделями
        try:
            self.cursor.execute(f"""
                create table if not exists models_alpha_4
                (
                    model_id serial not null,
                    angle smallint not null,
                    pressure_coefficients smallint[][] not null,
                    
                    constraint FK_{str(uuid.uuid4()).replace('-', '')} foreign key (model_id) 
                    references experiments_alpha_4(model_id)
                )""")
            self.__connection.commit()
            print(f"Создана таблица с моделями с параметром 4")
        except Exception as error:
            print(f"Ошибка -> {error}")
            self.__connection.commit()

        try:
            self.cursor.execute(f"""
                create table if not exists models_alpha_6
                (
                    model_id serial not null,
                    angle smallint not null,
                    pressure_coefficients smallint[][] not null,

                    constraint FK_{str(uuid.uuid4()).replace('-', '')} foreign key (model_id) 
                    references experiments_alpha_6(model_id)
                )""")
            self.__connection.commit()
            print(f"Создана таблица с моделями с параметром 6")

        except Exception as error:
            print(f"Ошибка -> {error}")
            self.__connection.commit()

    def generate_not_exists_case(self, alpha, model_name, angle):
        """Определяет параметры для генерация несуществующего варианта"""
        print(f'Генерация модели {model_name} с параметром {alpha} и углом {angle}')
        if model_name[0] == model_name[1]:  # в основании квадрат
            if any([45 < angle < 90, 135 < angle < 180, 225 < angle < 270, 315 < angle < 360]):
                self.__extrapolatedAnglesInfoList[f'T{model_name}_{alpha}_{angle:03d}'] = self.generator('rev',
                                                                                                         alpha,
                                                                                                         model_name,
                                                                                                         angle)
            else:
                self.__extrapolatedAnglesInfoList[f'T{model_name}_{alpha}_{angle:03d}'] = self.generator('for',
                                                                                                         alpha,
                                                                                                         model_name,
                                                                                                         angle)
        else:  # в основании прямоугольник
            if any([90 < angle < 180, 270 < angle < 360]):
                self.__extrapolatedAnglesInfoList[f'T{model_name}_{alpha}_{angle:03d}'] = self.generator('rev',
                                                                                                         alpha,
                                                                                                         model_name,
                                                                                                         angle)
            else:
                self.__extrapolatedAnglesInfoList[f'T{model_name}_{alpha}_{angle:03d}'] = self.generator('for',
                                                                                                         alpha,
                                                                                                         model_name,
                                                                                                         angle)

    def generator(self, type_gen, alpha, model_name, angle):
        """Определяет как менять расстановку датчиков и вызывает расстановщика"""
        angle_parent = {
            'rev': 90 * (angle // 90 + 1) - angle,
            'for': angle % 90
        }

        # определение параметров для перестановки данных
        type_view = {
            True: {
                'rev': {
                    45 < angle < 90: (1, 0, 3, 2),
                    135 < angle < 180: (2, 1, 0, 3),
                    225 < angle < 270: (3, 2, 1, 0),
                    315 < angle < 360: (0, 3, 1, 2)
                },
                'for': {
                    0 <= angle <= 45: (0, 1, 2, 3),
                    90 <= angle <= 135: (3, 0, 1, 2),
                    180 <= angle <= 225: (2, 3, 0, 1),
                    270 <= angle <= 315: (1, 2, 3, 0)
                }},
            False: {
                'rev': {
                    90 < angle < 180: (2, 1, 0, 3),
                    270 < angle < 360: (0, 3, 2, 1),
                },
                'for': {
                    0 <= angle <= 90: (0, 1, 2, 3),
                    180 <= angle <= 270: (2, 3, 0, 1)
                }}
        }
        # model_name[0] == model_name[1] проверка на квадрат в основании
        view_data = type_view[model_name[0] == model_name[1]][type_gen][True]
        return self.changer_sequence_sensors(alpha, model_name, angle_parent[type_gen], view_data, type_gen)

    def changer_sequence_sensors(self, alpha, model_name, angle, view_data = (0, 1, 2, 3), type_gen = 'for'):
        """Меняет порядок следования датчиков и координат тем самым генерируя не существующие углы"""
        coeffs = self.get_pressure_coefficients(alpha, model_name, angle)
        x, z = self.get_coordinates(alpha, model_name)

        f1, f2, f3, f4 = view_data
        count_sensors_on_middle = int(model_name[0]) * 5
        count_sensors_on_side = int(model_name[1]) * 5
        count_sensors_on_model = len(coeffs[0])
        count_row = count_sensors_on_model // (2 * (count_sensors_on_middle + count_sensors_on_side))

        if type_gen == 'for':
            for i in range(len(coeffs)):
                arr = np.array(coeffs[i]).reshape((count_row, -1))
                arr = np.split(arr, [count_sensors_on_middle,
                                     count_sensors_on_middle + count_sensors_on_side,
                                     2 * count_sensors_on_middle + count_sensors_on_side,
                                     2 * (count_sensors_on_middle + count_sensors_on_side)
                                     ], axis=1)

                coeffs[i] = np.concatenate((arr[f1],
                                            arr[f2],
                                            arr[f3],
                                            arr[f4]), axis=1).reshape(count_sensors_on_model)
            arr = np.array(x).reshape((count_row, -1))
            arr = np.split(arr, [count_sensors_on_middle,
                                 count_sensors_on_middle + count_sensors_on_side,
                                 2 * count_sensors_on_middle + count_sensors_on_side,
                                 2 * (count_sensors_on_middle + count_sensors_on_side)
                                 ], axis=1)

            x = np.concatenate((arr[f1],
                                arr[f2],
                                arr[f3],
                                arr[f4]), axis=1).reshape(count_sensors_on_model)

        else:
            for i in range(len(coeffs)):
                arr = np.array(coeffs[i]).reshape((count_row, -1))
                arr = np.split(arr, [count_sensors_on_middle,
                                     count_sensors_on_middle + count_sensors_on_side,
                                     2 * count_sensors_on_middle + count_sensors_on_side,
                                     2 * (count_sensors_on_middle + count_sensors_on_side)
                                     ], axis=1)

                coeffs[i] = np.concatenate((np.flip(arr[f1], axis=1),
                                            np.flip(arr[f2], axis=1),
                                            np.flip(arr[f3], axis=1),
                                            np.flip(arr[f4], axis=1)), axis=1).reshape(count_sensors_on_model)
            arr = np.array(x).reshape((count_row, -1))
            arr = np.split(arr, [count_sensors_on_middle,
                                 count_sensors_on_middle + count_sensors_on_side,
                                 2 * count_sensors_on_middle + count_sensors_on_side,
                                 2 * (count_sensors_on_middle + count_sensors_on_side)
                                 ], axis=1)

            x = np.concatenate((np.flip(arr[f1], axis=1),
                                np.flip(arr[f2], axis=1),
                                np.flip(arr[f3], axis=1),
                                np.flip(arr[f4], axis=1)), axis=1).reshape(count_sensors_on_model)
        return coeffs, x, z

    def get_coordinates(self, alpha, model_name):
        """Возвращает координаты датчиков"""
        if alpha == '4' or alpha == 4:
            self.cursor.execute("""
                        select x_coordinates, z_coordinates
                        from experiments_alpha_4
                        where model_name = (%s)
                    """, (model_name,))

        elif alpha == '6' or alpha == 6:
            self.cursor.execute("""
                        select x_coordinates, z_coordinates
                        from experiments_alpha_6
                        where model_name = (%s)
                    """, (model_name,))
        self.__connection.commit()
        x, z = self.cursor.fetchall()[0]
        return x, z

    def get_pressure_coefficients(self, alpha, model_name, angle):
        """Возвращает коэффициенты давления"""
        if alpha == '4' or alpha == 4:
            self.cursor.execute("""
                        select pressure_coefficients
                        from models_alpha_4
                        where model_id = (
                        select model_id
                        from experiments_alpha_4
                        where model_name = (%s)
                        ) and angle = (%s)
                    """, (model_name, angle))

        elif alpha == '6' or alpha == 6:
            self.cursor.execute("""
                        select pressure_coefficients
                        from models_alpha_6
                        where model_id = (
                        select model_id
                        from experiments_alpha_6
                        where model_name = (%s)
                        ) and angle = (%s)
                    """, (model_name, angle))
        self.__connection.commit()
        pressure_coefficients = self.cursor.fetchall()
        if not pressure_coefficients:
            if model_name[0] == model_name[1] and angle <= 45 or model_name[0] != model_name[1] and angle <= 90:
                print(f"T{model_name}_{alpha}_{angle:03d} не существует и не может быть сгенерирован")
                return None
            else:
                self.generate_not_exists_case(alpha, model_name, angle)
                return self.__extrapolatedAnglesInfoList[f'T{model_name}_{alpha}_{angle:03d}'][0]
        return pressure_coefficients[0][0]

    def graphs(self, mode, alpha, model_name, angle):
        """Метод отвечает за вызов графика из класса Artist"""
        angle = int(angle) % 360
        if angle % 5 != 0:
            print('Углы должны быть кратны 5')
            return None
        pressure_coefficients = np.array(self.get_pressure_coefficients(alpha, model_name, angle)) / 1000
        try:
            coordinates = self.__extrapolatedAnglesInfoList[f'T{model_name}_{alpha}_{angle:03d}'][1:]
        except:
            coordinates = self.get_coordinates(alpha, model_name)

        return Artist.isofield(mode, pressure_coefficients, coordinates, alpha, model_name, angle)

    def generate_signal(self, alpha, model_name, angle, x_gen, z_gen):
        new_signal = []
        pressure_coefficients = np.array(self.get_pressure_coefficients(alpha, model_name, angle)) / 1000
        x, z = self.get_coordinates(alpha, model_name)
        breadth, depth, height = int(model_name[0]) / 10, int(model_name[1]) / 10, int(model_name[2]) / 10
        count_sensors_on_model = len(pressure_coefficients[0])
        count_sensors_on_middle = int(model_name[0]) * 5
        count_sensors_on_side = int(model_name[1]) * 5
        count_row = count_sensors_on_model // (2 * (count_sensors_on_middle + count_sensors_on_side))
        face = -1
        if 0 <= x_gen <= breadth:
            face = 0
        elif breadth < x_gen <= (breadth + depth):
            face = 1
            x_gen -= breadth

        elif (breadth + depth) < x_gen <= (2 * breadth + depth):
            face = 2
            x_gen -= (breadth + depth)

        else:
            face = 3
            x_gen -= (2 * breadth + depth)

        x = np.reshape(x, (count_row, -1))
        x = np.split(x, [count_sensors_on_middle,
                         count_sensors_on_middle + count_sensors_on_side,
                         2 * count_sensors_on_middle + count_sensors_on_side,
                         2 * (count_sensors_on_middle + count_sensors_on_side)
                         ], axis=1)

        z = np.reshape(z, (count_row, -1))
        z = np.split(z, [count_sensors_on_middle,
                         count_sensors_on_middle + count_sensors_on_side,
                         2 * count_sensors_on_middle + count_sensors_on_side,
                         2 * (count_sensors_on_middle + count_sensors_on_side)
                         ], axis=1)

        del x[4]
        del z[4]

        for i in range(4):
            x[i] = x[i].tolist()
            z[i] = z[i].tolist()

        # x1...x4 координаты отдельных граней
        x1, x2, x3, x4 = x  # x это тензор со всеми координатами граней по ширине
        # z1...z4 координаты отдельных граней
        z1, z2, z3, z4 = z  # z это тензор со всеми координатами граней по высоте

        x = [np.array(x1), np.array(x2), np.array(x3), np.array(x4)]  # Входные координаты для изополей
        z = [np.array(z1), np.array(z2), np.array(z3), np.array(z4)]  # Входные координаты для изополей

        # x это координаты по ширине
        x_old = x[face].reshape(1, -1)[0]
        # Вычитаем чтобы все координаты по x находились в интервале [0, 1]
        if face == 1:
            x_old -= breadth
        elif face == 2:
            x_old -= (breadth + depth)
        elif face == 3:
            x_old -= (2 * breadth + depth)
        # z это координаты по высоте
        z_old = z[face].reshape(1, -1)[0]
        coords = [[i1, j1] for i1, j1 in zip(x_old, z_old)]  # Старые координаты

        for ind in range(len(pressure_coefficients)):
            coefficient = pressure_coefficients[ind]

            coefficient = np.reshape(coefficient, (count_row, -1))
            coefficient = np.split(coefficient, [count_sensors_on_middle,
                                                 count_sensors_on_middle + count_sensors_on_side,
                                                 2 * count_sensors_on_middle + count_sensors_on_side,
                                                 2 * (
                                                         count_sensors_on_middle + count_sensors_on_side)
                                                 ], axis=1)
            del coefficient[4]
            data_old = coefficient[face].reshape(1, -1)[0]
            # Интерполятор полученный на основе имеющихся данных
            interpolator = Artist.interpolator(coords, data_old)
            new_signal.append(float(interpolator([[x_gen, z_gen]])))

        return new_signal

    def graphic_mean(self, alpha, model_name, angle):
        pr_norm = np.array(self.get_pressure_coefficients(alpha, model_name, angle)) / 1000
        count_sensors_on_model = len(pr_norm[0])
        count_sensors_on_middle = int(model_name[0]) * 5
        count_sensors_on_side = int(model_name[1]) * 5
        count_row = count_sensors_on_model // (2 * (count_sensors_on_middle + count_sensors_on_side))

        sum_int_x = []
        sum_int_y = []

        for coeff in pr_norm:
            coeff = np.reshape(coeff, (count_row, -1))
            coeff = np.split(coeff, [count_sensors_on_middle,
                                     count_sensors_on_middle + count_sensors_on_side,
                                     2 * count_sensors_on_middle + count_sensors_on_side,
                                     2 * (count_sensors_on_middle + count_sensors_on_side)
                                     ], axis=1)
            del coeff[4]
            faces_x = []
            faces_y = []
            for face in range(len(coeff)):
                if face in [0, 2]:
                    faces_x.append(np.sum(coeff[face]) / (count_sensors_on_model / 4))
                else:
                    faces_y.append(np.sum(coeff[face]) / (count_sensors_on_model / 4))

            sum_int_x.append((faces_x[0] - faces_x[1]))
            sum_int_y.append((faces_y[0] - faces_y[1]))
        print(f"""
                среднее по Cy
                {sum(sum_int_y) / 32768}
                среднее по Cx
                {sum(sum_int_x) / 32768}
                """)
        print(f"""
                стандартное откл по Cx
                {np.std(sum_int_x)}
                стандартное откл по Cy
                {np.std(sum_int_y)}
                """)
        fig, graph = plt.subplots(1, 2, figsize=(16, 9))
        graph[0].plot(list(range(1, 32769)), sum_int_x, label='Cx')
        graph[1].plot(list(range(1, 32769)), sum_int_y, label='Cy')
        for i in range(2):
            graph[i].legend()
            graph[i].grid()
        plt.savefig(f'{folder_out}\\график средних {model_name}_{alpha}')
        plt.clf()
        plt.close()

    def spectr_mean(self, alpha, model_name, angle = 0, border = 30000):
        pr_norm = np.array(self.get_pressure_coefficients(alpha, model_name, angle)) / 1000
        count_sensors_on_model = len(pr_norm[0])
        count_sensors_on_middle = int(model_name[0]) * 5
        count_sensors_on_side = int(model_name[1]) * 5
        count_row = count_sensors_on_model // (2 * (count_sensors_on_middle + count_sensors_on_side))

        sum_int_x = []
        sum_int_y = []

        for coeff in pr_norm:
            coeff = np.reshape(coeff, (count_row, -1))
            coeff = np.split(coeff, [count_sensors_on_middle,
                                     count_sensors_on_middle + count_sensors_on_side,
                                     2 * count_sensors_on_middle + count_sensors_on_side,
                                     2 * (count_sensors_on_middle + count_sensors_on_side)
                                     ], axis=1)
            del coeff[4]
            faces_x = []
            faces_y = []
            for face in range(len(coeff)):
                if face in [0, 2]:
                    faces_x.append(np.sum(coeff[face]) / (count_sensors_on_model / 4))
                else:
                    faces_y.append(np.sum(coeff[face]) / (count_sensors_on_model / 4))

            sum_int_x.append((faces_x[0] - faces_x[1]))
            sum_int_y.append((faces_y[0] - faces_y[1]))
        N = len(sum_int_x)

        yfCx = (1 / N) * (np.abs(fft(sum_int_x)))[1:N // 2]
        yfCy = (1 / N) * (np.abs(fft(sum_int_y)))[1:N // 2]

        FD = 1000
        xf = rfftfreq(N, 1 / FD)[1:N // 2]

        fig, graph = plt.subplots(1, 2, figsize=(16, 9))
        if border == -1:
            border = 100000

        graph[0].plot(xf[:border], yfCx[:border], antialiased=True, label='Cx')
        graph[1].plot(xf[:border], yfCy[:border], antialiased=True, label='Cy')
        for i in range(2):
            graph[i].legend()
            graph[i].grid()

        plt.savefig(f'{folder_out}\\спектр средних {model_name}_{alpha}')
        plt.clf()
        plt.close()

    def generate_model_pic(self, alpha, model_name):
        breadth, depth, height = int(model_name[0]) / 10, int(model_name[1]) / 10, int(model_name[2]) / 10
        size_x = 2 * (breadth + depth)
        x, z = self.get_coordinates(alpha, model_name)
        pr_coeff = np.array(self.get_pressure_coefficients(alpha, model_name, 0)) / 1000
        count_sensors = len(pr_coeff[0])
        fig, ax = plt.subplots(figsize=(7, 6), dpi=200, num=1, clear=True)
        ax.set_title('Channels position', fontweight='semibold', fontsize=8)
        ax.set_xlabel('Horizontal Direction /m', fontweight='semibold', fontsize=8)
        ax.set_ylabel('Vertical Direction /m', fontweight='semibold', fontsize=8)
        fig.text(0.1, 0.005,
                 f'Model geometrical parameters of a high-rise building: H={height}m, B={breadth}m, D={depth}m, '
                 f'Model scal=1\\{alpha}00', fontweight='semibold', fontsize=8)
        ax.set_ylim(0, height)
        ax.set_xlim(0, size_x)
        xtick_v = 0.05
        ytick_v = 0.02 if height in [0.1, 0.2] else 0.05
        xticks = np.arange(0, size_x + xtick_v, xtick_v)
        yticks = np.arange(0, height + ytick_v, ytick_v)
        xlabels = ['0'] + [str(i)[:4].rstrip('0') for i in xticks[1:]]
        ylabels = ['0'] + [str(i)[:4].rstrip('0') for i in yticks[1:]]
        ax.set_xticks(xticks, labels=xlabels)
        ax.set_yticks(yticks, labels=ylabels)
        for i in range(1, int(size_x * 10)):
            ax.plot([i / 10, i / 10], [0, height], linestyle='--', color='black')
        ax.plot(x, z, '+')
        for i, j, text in zip(x, z, [str(i) for i in range(1, count_sensors + 1)]):
            ax.text(i, j - 0.01, text, fontsize=8)
        ax.set_aspect('equal') if height == 0.1 else None

        if not os.path.isdir(f'{os.getcwd()}\\Отчет {model_name}_{alpha}'):
            os.mkdir(f'{os.getcwd()}\\Отчет {model_name}_{alpha}')

        plt.savefig(f'Отчет {model_name}_{alpha}\\Модель {model_name}_{alpha}.png')

    def generate_envelopes(self, alpha, model_name):
        if model_name[0] == model_name[1]:
            border = 50
        else:
            border = 95

        for angle in range(0, border, 5):
            pr_coeff = np.array(self.get_pressure_coefficients(alpha, model_name, angle)) / 1000
            mean_pr = np.mean(pr_coeff, axis=0).round(4)
            rms_pr = np.array([np.sqrt(i.dot(i) / i.size) for i in pr_coeff.T]).round(4)
            std_pr = np.std(pr_coeff, axis=0).round(4)
            max_pr = np.max(pr_coeff, axis=0).round(4)
            min_pr = np.min(pr_coeff, axis=0).round(4)

            count_sensors = len(pr_coeff[0])
            fig, ax = plt.subplots(figsize=(12, 6), dpi=200, num=1, clear=True)
            ox = [i for i in range(1, count_sensors + 1)]
            ax.plot(ox, mean_pr, '-', label='MEAN')
            ax.plot(ox, rms_pr, '-', label='RMS')
            ax.plot(ox, std_pr, '-', label='STD')
            ax.plot(ox, max_pr, '-', label='MAX')
            ax.plot(ox, min_pr, '-', label='MIN')

            xticks = np.arange(0, count_sensors + 20, 20)
            yticks = np.arange(np.min(min_pr), np.max(max_pr) + 0.2, 0.2).round(2)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.set_xlim(0, count_sensors + 1)
            ax.set_ylim(np.min(yticks), np.max(yticks))
            ax.legend()
            ax.grid()
            plt.savefig(f'Отчет {model_name}_{alpha}\\Огибающие {model_name}_{alpha} {angle:02}.png')

    def get_face_number(self, alpha, model_name):
        if alpha == '4' or alpha == 4:
            self.cursor.execute("""
                        select face_number
                        from experiments_alpha_4
                        where model_name = (%s)
                    """, (model_name,))

        elif alpha == '6' or alpha == 6:
            self.cursor.execute("""
                        select face_number
                        from experiments_alpha_6
                        where model_name = (%s)
                    """, (model_name,))
        self.__connection.commit()
        return self.cursor.fetchall()[0][0]

    def get_uh_average_wind_speed(self, alpha, model_name):
        if alpha == '4' or alpha == 4:
            self.cursor.execute("""
                        select uh_averagewindspeed
                        from experiments_alpha_4
                        where model_name = (%s)
                    """, (model_name,))

        elif alpha == '6' or alpha == 6:
            self.cursor.execute("""
                        select uh_averagewindspeed
                        from experiments_alpha_6
                        where model_name = (%s)
                    """, (model_name,))
        self.__connection.commit()
        return self.cursor.fetchall()[0][0]

    def get_info_sensors(self, alpha, model_name):
        info_sensors = []
        x, z = self.get_coordinates(alpha, model_name)
        pr_coeff = np.array(self.get_pressure_coefficients(alpha, model_name, 0)) / 1000
        face_number = self.get_face_number(alpha, model_name)
        breadth, depth, height = int(model_name[0]) / 10, int(model_name[1]) / 10, int(model_name[2]) / 10
        sensors_on_model = len(pr_coeff[0])
        x_new, y_new = self.converter_coordinates(x, depth, breadth, face_number, sensors_on_model)
        mean_pr = np.mean(pr_coeff, axis=0).round(4)
        rms_pr = np.array([np.sqrt(i.dot(i) / i.size) for i in pr_coeff.T]).round(4)
        std_pr = np.std(pr_coeff, axis=0).round(4)
        max_pr = np.max(pr_coeff, axis=0).round(4)
        min_pr = np.min(pr_coeff, axis=0).round(4)

        for i in range(sensors_on_model):
            row = [i + 1,
                   x_new[i],
                   y_new[i],
                   z[i],
                   mean_pr[i],
                   rms_pr[i],
                   std_pr[i],
                   max_pr[i],
                   min_pr[i],
                   np.max([np.abs(min_pr[i]), np.abs(max_pr[i])]).round(2),
                   (np.abs(max_pr[i] - mean_pr[i]) / std_pr[i]).round(2),
                   (np.abs(min_pr[i] - mean_pr[i]) / std_pr[i]).round(2)
                   ]
            info_sensors.append(row)
        return info_sensors

    def get_sum_coeff(self, alpha, model_name, angle = 0):
        sum_int_x = []
        sum_int_y = []
        pr_coeff = np.array(self.get_pressure_coefficients(alpha, model_name, angle)) / 1000
        count_sensors_on_model = len(pr_coeff[0])
        count_sensors_on_middle = int(model_name[0]) * 5
        count_sensors_on_side = int(model_name[1]) * 5
        count_row = count_sensors_on_model // (2 * (count_sensors_on_middle + count_sensors_on_side))
        for coeff in pr_coeff:

            coeff = np.reshape(coeff, (count_row, -1))
            coeff = np.split(coeff, [count_sensors_on_middle,
                                     count_sensors_on_middle + count_sensors_on_side,
                                     2 * count_sensors_on_middle + count_sensors_on_side,
                                     2 * (count_sensors_on_middle + count_sensors_on_side)
                                     ], axis=1)
            del coeff[4]
            faces_x = []
            faces_y = []
            for face in range(len(coeff)):
                if face in [0, 2]:
                    faces_x.append(np.sum(coeff[face]) / (count_sensors_on_model / 4))
                else:
                    faces_y.append(np.sum(coeff[face]) / (count_sensors_on_model / 4))

            sum_int_x.append((faces_x[0] - faces_x[1]))
            sum_int_y.append((faces_y[0] - faces_y[1]))
        return sum_int_x, sum_int_y

    def get_sum_cmz(self, alpha, model_name, angle = 0):
        sum_cmz = []
        pr_coeff = np.array(self.get_pressure_coefficients(alpha, model_name, angle)) / 1000
        breadth, depth, height = int(model_name[0]) / 10, int(model_name[1]) / 10, int(model_name[2]) / 10
        v2 = breadth
        v3 = breadth + depth
        v4 = 2 * breadth + depth
        mid13_x = breadth / 2
        mid24_x = depth / 2
        count_sensors_on_model = len(pr_coeff[0])
        count_sensors_on_middle = int(model_name[0]) * 5
        count_sensors_on_side = int(model_name[1]) * 5
        count_row = count_sensors_on_model // (2 * (count_sensors_on_middle + count_sensors_on_side))
        x, z = self.get_coordinates(alpha, model_name)

        x = np.reshape(x, (count_row, -1))
        x = np.split(x, [count_sensors_on_middle,
                         count_sensors_on_middle + count_sensors_on_side,
                         2 * count_sensors_on_middle + count_sensors_on_side,
                         2 * (count_sensors_on_middle + count_sensors_on_side)
                         ], axis=1)

        del x[4]
        x[1] -= v2
        x[2] -= v3
        x[3] -= v4
        mx = np.array([
            abs(x[0] - mid13_x),
            abs(x[1] - mid24_x),
            abs(x[2] - mid13_x),
            abs(x[3] - mid24_x),
        ])
        coeffs_norm_13 = [1 if i <= count_sensors_on_middle // 2 else -1 for i in range(count_sensors_on_middle)]
        coeffs_norm_24 = [1 if i <= count_sensors_on_side // 2 else -1 for i in range(count_sensors_on_side)]
        for coeff in pr_coeff:

            coeff = np.reshape(coeff, (count_row, -1))
            coeff = np.split(coeff, [count_sensors_on_middle,
                                     count_sensors_on_middle + count_sensors_on_side,
                                     2 * count_sensors_on_middle + count_sensors_on_side,
                                     2 * (count_sensors_on_middle + count_sensors_on_side)
                                     ], axis=1)
            del coeff[4]
            for i in range(4):
                if i in [0, 2]:
                    coeff[i] *= coeffs_norm_13
                else:
                    coeff[i] *= coeffs_norm_24
            cmz = mx * coeff
            sum_cmz.append(np.sum(cmz))

        return sum_cmz

    def graph_sum_coeff(self, alpha, model_name, angle = 0):
        info_sum_coeff = []
        sum_int_x, sum_int_y = self.get_sum_coeff(alpha, model_name, angle)
        sum_cmz = self.get_sum_cmz(alpha, model_name, angle)
        fig, ax = plt.subplots(figsize=(7, 6), dpi=200, num=1, clear=True)
        ox = np.linspace(0, 32.768, 32768)
        for data, name in zip((sum_int_x, sum_int_y, sum_cmz), ("CX", "CY", "CMz")):
            ax.plot(ox, data, label=name)
            ax.legend()
            ax.grid()
            ax.set_xlim(0, 32.768)
            plt.savefig(f'Отчет {model_name}_{alpha}\\{name} {model_name}_{alpha}.png')
            ax.clear()
            max_pr = np.max(data).round(2)
            mean_pr = np.mean(data).round(2)
            min_pr = np.min(data).round(2)
            std_pr = np.std(data).round(2)
            info_sum_coeff.append([
                name,
                mean_pr,
                np.sqrt(np.array(data).dot(np.array(data)) / np.array(data).size).round(2),
                std_pr,
                max_pr,
                min_pr,
                np.max([np.abs(min_pr), np.abs(max_pr)]).round(2),
                (np.abs(max_pr - mean_pr) / std_pr).round(2),
                (np.abs(min_pr - mean_pr) / std_pr).round(2)
            ])

        return info_sum_coeff

    def spectr_sum_report(self, alpha, model_name, angle = 0):
        sum_int_x, sum_int_y = self.get_sum_coeff(alpha, model_name, angle)
        sum_cmz = self.get_sum_cmz(alpha, model_name, angle)
        N = len(sum_int_x)

        yfCx = (1 / N) * (np.abs(fft(sum_int_x)))[1:N // 2]
        yfCy = (1 / N) * (np.abs(fft(sum_int_y)))[1:N // 2]
        yfCMz = (1 / N) * (np.abs(fft(sum_cmz)))[1:N // 2]

        FD = 1000
        xf = rfftfreq(N, 1 / FD)[1:N // 2]

        border = 500
        count_peaks = 2

        fig, ax = plt.subplots(figsize=(12, 6), dpi=200, num=1, clear=True)

        for data, name in zip([yfCx, yfCy, yfCMz], ['Cx', 'Cy', 'CMz']):
            ax.plot(xf[:border], data[:border], antialiased=True, label=name)
            peaks = np.sort(data[:border])[::-1][:count_peaks]

            for peak in range(count_peaks):
                x = xf[:border][np.where(data[:border] == peaks[peak])].round(3)[0]
                y = peaks[peak]
                ax.annotate(x, xy=(x, y))

            ax.legend()
            ax.grid()
            plt.savefig(f'Отчет {model_name}_{alpha}\\Спектр {name} {model_name}_{alpha}.png')
            ax.clear()

    def spect_sum_st(self, alpha, model_name, angle, parameters):
        labels = parameters['labels']
        data_sum = parameters['data']
        data_spectr = []
        fi_st = []
        N = len(data_sum[0])
        for i in data_sum:
            data_spectr.append((1 / N) * (np.abs(fft(i)))[1:N // 2])

        FD = 1000
        xf = rfftfreq(N, 1 / FD)[1:N // 2]

        border = 500
        count_peaks = 1

        fig, ax = plt.subplots(figsize=(12, 6), dpi=200, num=1, clear=True)

        for data, name in zip(data_spectr, labels):
            ax.plot(xf[:border], data[:border], antialiased=True, label=name)
            peaks = np.sort(data[:border])[::-1][:count_peaks]

            for peak in range(count_peaks):
                x = xf[:border][np.where(data[:border] == peaks[peak])].round(3)[0]
                y = peaks[peak]
                ax.annotate(x, xy=(x, y))
                fi_st.append(x)

        ax.legend()
        ax.grid()
        plt.savefig(f'Отчет {model_name}_{alpha}\\Спектр суммарных сил {model_name}_{alpha}_{angle}.png')
        return fi_st

    def get_info_sum_coeff_st(self, alpha, model_name):
        info_sum_coeff_st = []
        speed = self.get_uh_average_wind_speed(alpha, model_name)
        size = float(model_name[0]) / 10
        for angle in range(0, 50, 5):
            sum_x, sum_y = self.get_sum_coeff(alpha, model_name, angle=angle)
            sum_cmz = self.get_sum_cmz(alpha, model_name, angle=angle)
            fi_st = self.spect_sum_st(alpha, model_name, angle, {'labels': ('Cx', 'CY', 'CMz'),
                                                                 'data': (sum_x, sum_y, sum_cmz)})
            fi_st = (np.array(fi_st) * size / speed).round(4).tolist()
            info_sum_coeff_st.append([angle, *fi_st])
        return info_sum_coeff_st

    def get_st(self, alpha, model_name, count_sensors):
        fi = []
        speed = self.get_uh_average_wind_speed(alpha, model_name)
        size = float(model_name[0]) / 10
        N = 32768
        FD = 1000
        border = 500
        count_peaks = 1
        for angle in range(0, 50, 5):
            filist = []

            pr_coeff = np.array(self.get_pressure_coefficients(alpha, model_name, angle)).T / 1000

            fig, ax = plt.subplots(figsize=(12, 6), dpi=200, num=1, clear=True)
            for i in range(count_sensors):

                yf = (1 / N) * (np.abs(fft(pr_coeff[i])))[1:N // 2]
                xf = rfftfreq(N, 1 / FD)[1:N // 2]
                ax.plot(xf[:border], yf[:border], antialiased=True, label=i + 1)

                peaks = np.sort(yf[:border])[::-1][:count_peaks]

                for peak in range(count_peaks):
                    x = xf[:border][np.where(yf[:border] == peaks[peak])].round(3)[0]
                    filist.append(x)
            fi.append((np.array(filist) * size / speed).round(4))
            # ax.legend()
            ax.grid()
            plt.savefig(f'Отчет {model_name}_{alpha}\\числа Струхаля для датчиков {model_name}_{alpha}_{angle}.png')
        return np.array(fi).T

    def get_info_sens_st(self, alpha, model_name):
        breadth, depth = int(model_name[0]) / 10, int(model_name[1]) / 10
        info_sens_st = []
        x, z = self.get_coordinates(alpha, model_name)
        face_number = self.get_face_number(alpha, model_name)
        x, y = self.converter_coordinates(x, depth, breadth, face_number, len(face_number))
        st = self.get_st(alpha, model_name, len(face_number))
        for i in range(len(face_number)):
            info_sens_st.append([i + 1, x[i], y[i], z[i], *st[i]])
        return info_sens_st

    def generate_report(self, alpha, model_name):
        counter = 0
        counter_plots = 0
        doc = Document()
        title = doc.add_paragraph().add_run(f'Отчет {model_name}_{alpha}')
        doc.paragraphs[counter].alignment = WD_ALIGN_PARAGRAPH.CENTER
        title.font.size = Pt(24)
        ####################### 1. Параметры здания #######################
        doc.add_paragraph().add_run('1. Параметры здания').font.size = Pt(15)
        counter += 1
        doc.paragraphs[counter].alignment = WD_ALIGN_PARAGRAPH.CENTER
        self.generate_model_pic(alpha, model_name)
        doc.add_picture(f'Отчет {model_name}_{alpha}\\Модель {model_name}_{alpha}.png')
        counter += 1
        ####################### 2. Статистика по датчиках. Максимумы и огибающие #######################
        doc.add_paragraph().add_run('2. Статистика по датчиках. Максимумы и огибающие').font.size = Pt(15)
        counter += 1
        doc.paragraphs[counter].alignment = WD_ALIGN_PARAGRAPH.CENTER
        self.generate_envelopes(alpha, model_name)
        envelopes = glob.glob(f'Отчет {model_name}_{alpha}\\Огибающие *.png')
        for env in envelopes:
            doc.add_picture(env)
            counter += 1
            angle = env[-6:-4]
            doc.paragraphs[counter].add_run(f'\nОгибающие {model_name}_{alpha} угол {angle}').font.size = Pt(12)
            counter_plots += 1
        ####################### 3. Статистика по датчикам в табличном виде #######################
        doc.add_paragraph().add_run('3. Статистика по датчикам в табличном виде ').font.size = Pt(15)
        counter += 1
        doc.paragraphs[counter].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.paragraphs[counter].add_run(
            f'\nТаблица 1. ТПУ {model_name}_{alpha}, RUMB=0 Аэродинамический коэффициент в датчиках').font.size = Pt(12)

        header_1 = (
            'ДАТЧИК',
            'X(мм)',
            'Y(мм)',
            'Z(мм)',
            'СРЕДНЕЕ',
            'RMS',
            'СТАНДАРТНОЕ ОТКЛОНЕНИЕ',
            'МАКСИМУМ',
            'МИНИМУМ',
            'РАСЧЕТНОЕ',
            'ОБЕСП+',
            'ОБЕСП-'
        )

        table_1 = doc.add_table(rows=1, cols=len(header_1))
        table_1.style = 'Table Grid'
        hdr_cells = table_1.rows[0].cells
        for i in range(len(header_1)):
            hdr_cells[i].add_paragraph().add_run(header_1[i]).font.size = Pt(8)

        info_sensors = self.get_info_sensors(alpha, model_name)

        for rec in info_sensors:
            row_cells = table_1.add_row().cells
            for i in range(len(rec)):
                row_cells[i].add_paragraph().add_run(str(rec[i])).font.size = Pt(8)
        ####################### 4. Суммарные значения аэродинамических коэффициентов #######################
        doc.add_paragraph().add_run('4. Суммарные значения аэродинамических коэффициентов').font.size = Pt(15)
        counter += 1
        doc.paragraphs[counter].alignment = WD_ALIGN_PARAGRAPH.CENTER
        info_sum_coeff = self.graph_sum_coeff(alpha, model_name)
        for name_sum in ('CX', 'CY', 'CMz'):
            doc.add_picture(f'Отчет {model_name}_{alpha}\\{name_sum} {model_name}_{alpha}.png')
            counter += 1
            counter_plots += 1

        doc.paragraphs[counter].add_run(
            f'''\nТаблица 2. ТПУ {model_name}_{alpha}, 
                RUMB=0 Аэродинамические коэффициенты сил и моментов''').font.size = Pt(12)

        header_2 = (
            'СИЛА',
            'СРЕДНЕЕ',
            'RMS',
            'СТАНДАРТНОЕ ОТКЛОНЕНИЕ',
            'МАКСИМУМ',
            'МИНИМУМ',
            'РАСЧЕТНОЕ',
            'ОБЕСП+',
            'ОБЕСП-'
        )
        table_2 = doc.add_table(rows=1, cols=len(header_2))
        table_2.style = 'Table Grid'
        hdr_cells = table_2.rows[0].cells
        for i in range(len(header_2)):
            hdr_cells[i].add_paragraph().add_run(header_2[i]).font.size = Pt(8)

        for rec in info_sum_coeff:
            row_cells = table_2.add_row().cells
            for i in range(len(rec)):
                row_cells[i].add_paragraph().add_run(str(rec[i])).font.size = Pt(8)
        ####################### 5. Числа Струхаля суммарных сил #######################
        doc.add_paragraph().add_run('5. Числа Струхаля суммарных сил').font.size = Pt(15)
        counter += 1
        info_sum_coeff_st = self.get_info_sum_coeff_st(alpha, model_name)
        doc.paragraphs[counter].alignment = WD_ALIGN_PARAGRAPH.CENTER
        header_3 = (
            'fi',
            'St(CX)',
            'St(CY)',
            'St(CMz)'
        )
        table_3 = doc.add_table(rows=1, cols=len(header_3))
        table_3.style = 'Table Grid'
        hdr_cells = table_3.rows[0].cells
        for i in range(len(header_3)):
            hdr_cells[i].add_paragraph().add_run(header_3[i]).font.size = Pt(8)
        for rec in info_sum_coeff_st:
            row_cells = table_3.add_row().cells
            for i in range(len(rec)):
                row_cells[i].add_paragraph().add_run(str(rec[i])).font.size = Pt(8)
        for angle in range(0, 50, 5):
            doc.add_picture(f'Отчет {model_name}_{alpha}\\Спектр суммарных сил {model_name}_{alpha}_{angle}.png')
            counter += 1
            doc.paragraphs[counter].add_run(f'\nРисунок {counter_plots}. Угол {angle}.').font.size = Pt(12)
            counter_plots += 1
        ####################### 6. Числа Струхаля давлений датчиков #######################
        doc.add_paragraph().add_run('6. Числа Струхаля давлений датчиков').font.size = Pt(15)
        counter += 1
        doc.paragraphs[counter].alignment = WD_ALIGN_PARAGRAPH.CENTER
        header_4 = (
            'ДАТЧИК',
            'x',
            'y',
            'z',
            '0',
            '5',
            '10',
            '15',
            '20',
            '25',
            '30',
            '35',
            '40',
            '45'
        )
        table_4 = doc.add_table(rows=1, cols=len(header_4))
        table_4.style = 'Table Grid'
        hdr_cells = table_4.rows[0].cells
        for i in range(len(header_4)):
            hdr_cells[i].add_paragraph().add_run(header_4[i]).font.size = Pt(8)

        info_sens_st = self.get_info_sens_st(alpha, model_name)
        for angle in range(0, 50, 5):
            doc.add_picture(f'Отчет {model_name}_{alpha}\\числа Струхаля для датчиков {model_name}_{alpha}_{angle}.png')
            counter += 1
            doc.paragraphs[counter].add_run(
                f'\nРисунок {counter_plots}. Угол {angle} числа Струхаля для датчиков').font.size = Pt(12)
            counter_plots += 1
        for rec in info_sens_st:
            row_cells = table_4.add_row().cells
            for i in range(len(rec)):
                row_cells[i].add_paragraph().add_run(str(rec[i])).font.size = Pt(8)

        ####################### 7. Спектры #######################
        self.spectr_sum_report(alpha, model_name)
        doc.add_paragraph().add_run('7. Спектры').font.size = Pt(15)
        counter += 1
        doc.paragraphs[counter].alignment = WD_ALIGN_PARAGRAPH.CENTER
        for name in ('Cx', 'Cy', 'CMz'):
            doc.add_picture(f'Отчет {model_name}_{alpha}\\Спектр {name} {model_name}_{alpha}.png')
            counter += 1
            doc.paragraphs[counter].add_run(f'\nСпектр {name} {model_name}_{alpha}').font.size = Pt(12)

        doc.save(f'Отчет {model_name}_{alpha}\\Отчет {model_name}_{alpha}.docx')


if __name__ == '__main__':
    control = Controller()
    # пароль 08101430
    control.connect(database='tpu', password='2325070307')

    control.generate_report('4', '111')
    # control.get_info_sens_st('4', '111')
    control.disconnect()

    # while True:
    #     command = input("""
    #     1
    #         - изополя
    #         - спектры
    #         - средние
    #             - графики
    #             - значения
    #         - стандартное откл
    #     2
    #         - изополя
    #     3
    #         - графики средних
    #         - значения средних
    #         - стандартное откл
    #     4
    #         - спектры средних
    #
    #     1 например:
    #     1 114 6 30 700
    #     режим, параметр BDH, альфа, угол, до какого значения обрезать спектры
    #
    #     2 например
    #     2 113 4 0 std
    #     если нужно конкретное изополе ввести режим:
    #         - min
    #         - max
    #         - mean
    #         - std
    #     если нужны все, то all
    #     2 113 4 0 all
    #
    #     3 например
    #     3 111 4 0
    #
    #     4 например
    #     4 115 4 0 157
    #     чтобы отобразить весь спектр последний параметр должен быть -1
    #     """)
    #
    #     command = command.split()
    #
    #     try:
    #         global folder_out
    #         folder_out = f'режим {" ".join(map(str, command))}'
    #         os.mkdir(folder_out)
    #
    #     except:
    #         print('Уже существует')
    #         break
    #     model_name = command[1]
    # alpha = command[2]
    # angle = command[3]
    # match command[0]:
    #
    #     case '1':
    #         for mode in ('max', 'mean', 'min', 'std'):
    #             control.graphs(mode, alpha, model_name, angle)
    #         control.graphic_mean(alpha, model_name, angle)
    #         border = int(command[-1])
    #         control.spectr_mean(alpha, model_name, angle, border)
    #     case '2':
    #         if command[-1] == 'all':
    #             for mode in ('max', 'mean', 'min', 'std'):
    #                 control.graphs(mode, alpha, model_name, angle)
    #         else:
    #             control.graphs(command[-1], alpha, model_name, angle)
    #     case '3':
    #         control.graphic_mean(alpha, model_name, angle)
    #     case '4':
    #         border = int(command[-1])
    #         control.spectr_mean(alpha, model_name, angle, border)
    #
    # break
