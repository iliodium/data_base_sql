import os
import uuid
import random
import psycopg2
import numpy as np
import scipy.interpolate
import scipy.io as sio
import matplotlib.cm as cm
import matplotlib.tri as mtri
import matplotlib.pyplot as plt

from psycopg2 import Error
from scipy.fft import fft, rfftfreq
import matplotlib.pyplot as plt
import numpy as np


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
    def isofield(mode, pressure_coefficients, coordinates, integral, alpha, model_name, angle):
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
            'mean': 0.02 if alpha == 6 else 0.1,
            'min': 0.2,
            'std': 0.05,
        }  # Шаги для изополей
        #  Шаг для изополей и контурных линий
        levels = np.arange(np.round(min_v, 1), np.round(max_v, 1) + 0.1, steps[mode])
        x, z = np.array(coordinates)
        if integral[1] == 1:
            # Масштабирование
            print(model_name, 1.25)
            k = 1.25  # коэффициент масштабирования по высоте
            z *= k
        else:
            k = 1
            print(model_name, 1)
        breadth, depth, height = int(model_name[0]) / 10, int(model_name[1]) / 10, int(model_name[2]) / 10 * k
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
        data_old_integer = []  # данные для дискретного интегрирования по осям
        data_for_3d_model = []  # данные для 3D модели
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
            data_old_integer.append(data_old)
            coords = [[i1, j1] for i1, j1 in zip(x_old, z_old)]  # Старые координаты
            # Интерполятор полученный на основе имеющихся данных
            interpolator = Artist.interpolator(coords, data_old)
            integral_func.append(interpolator)
            # Получаем данные для несуществующих датчиков
            data_new = [float(interpolator([[X, Y]])) for X, Y in zip(x_new, z_new)]

            triang = mtri.Triangulation(x_new, z_new)
            refiner = mtri.UniformTriRefiner(triang)
            grid, value = refiner.refine_field(data_new, subdiv=4)
            data_for_3d_model.append((grid, value))
        #     data_colorbar = graph[i].tricontourf(grid, value, cmap=cmap, levels=levels, extend='both')
        #     aq = graph[i].tricontour(grid, value, linewidths=1, linestyles='solid', colors='black', levels=levels)
        #
        #     graph[i].clabel(aq, fontsize=13)
        #     graph[i].set_ylim([0, height])
        #     if breadth == depth == height:
        #         graph[i].set_aspect('equal')
        #     if i in [0, 2]:
        #         graph[i].set_xlim([0, breadth])
        #         graph[i].set_xticks(ticks=np.arange(0, breadth + 0.1, 0.1))
        #     else:
        #         graph[i].set_xlim([0, depth])
        #         graph[i].set_xticks(ticks=np.arange(0, depth + 0.1, 0.1))
        #     ret_int.append(interpolator)
        # fig.colorbar(data_colorbar, ax=graph, location='bottom', cmap=cmap, ticks=levels)
        # plt.show()
        plt.close()
        isofield_model().show(data_for_3d_model, levels)
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

    def graphs(self, graphic, mode, integtal, alpha, model_name, angle, new_signal):
        """Метод отвечает за вызов графика из класса Artist"""
        angle = int(angle) % 360
        if angle % 5 != 0:
            print('Углы должны быть кратны 5')
            return None
        modes_graphs = {
            'isofield_min': Artist.isofield_min,
            'isofield_mean': Artist.isofield_mean,
            'isofield_max': Artist.isofield_max,
            'isofield_std': Artist.isofield_std,
            'signal': Artist.signal,
            'spectrum': Artist.spectrum,
        }
        pressure_coefficients = np.array(self.get_pressure_coefficients(alpha, model_name, angle)) / 1000
        Artist.signal1(pressure_coefficients,new_signal)
        # pressure_coefficients1 = np.array(self.get_pressure_coefficients('4', '113', '0')) / 1000
        # try:
        #     coordinates = self.__extrapolatedAnglesInfoList[f'T{model_name}_{alpha}_{angle:03d}'][1:]
        # except:
        #     coordinates = self.get_coordinates(alpha, model_name)
        # # modes_graphs[mode](pressure_coefficients, pressure_coefficients1, alpha, model_name, angle)
        # return Artist.isofield(mode, pressure_coefficients, coordinates, integtal, alpha, model_name, angle)

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


# class isofield_model:
#
#     def show(self, data, levels):
#         import numpy as np
#         import matplotlib.pyplot as plt
#         from mpl_toolkits.mplot3d import Axes3D
#         import matplotlib.colors
#         ax = plt.figure().add_subplot(projection='3d')
#
#         for i in range(len(data)):
#             grid, value = data[i]
#             colors = plt.cm.get_cmap("jet", N - 1)(np.arange(N - 1))
#             cmap, norm = matplotlib.colors.from_levels_and_colors(levels, colors)
#             color_vals = cmap(norm(c))
#             if i == 0:
#                 # ax.tricontourf(grid, value,zdir='x', cmap='jet', levels=levels, extend='both')
#                 ax.tricontour(grid, value, zs=1, zdir='z', linewidths=1, linestyles='solid', colors='black',
#                               levels=levels)
#             # elif i == 1:
#             #     ax.tricontourf(grid, value, zdir='x', cmap='jet', levels=levels, extend='both')
#             #     ax.tricontour(grid, value, zdir='x', linewidths=1, linestyles='solid', colors='black',
#             #                   levels=levels)
#             # # elif i == 2:
#             #     ax.tricontourf(grid, value, zdir='y', cmap='jet', levels=levels, extend='both')
#             #     ax.tricontour(grid, value, zdir='y', linewidths=1, linestyles='solid', colors='black',
#             #                   levels=levels)
#             # else:
#             #     ax.tricontourf(grid, value, zdir='y', cmap='jet', levels=levels, extend='both')
#             #     ax.tricontour(grid, value, zdir='y', linewidths=1, linestyles='solid', colors='black',
#             #                   levels=levels)
#
#         # # Plot a sin curve using the x and y axes.
#         # x = np.linspace(0, 1, 100)
#         # y = np.sin(x * 2 * np.pi) / 2 + 0.5
#         # ax.plot(x, y, zs=0, zdir='x', label='curve in (x, y)1')
#         #
#         # x = np.linspace(0, 1, 100)
#         # y = np.sin(x * 2 * np.pi) / 2 + 0.5
#         # ax.plot(x, y, zs=1, zdir='x', label='curve in (x, y)1')
#         #
#         # x = np.linspace(0, 1, 100)
#         # y = np.sin(x * 2 * np.pi) / 2 + 0.5
#         # ax.plot(x, y, zs=0, zdir='y', label='curve in (x, y)2')
#         #
#         # x = np.linspace(0, 1, 100)
#         # y = np.sin(x * 2 * np.pi) / 2 + 0.5
#         # ax.plot(x, y, zs=1, zdir='y', label='curve in (x, y)2')
#
#         ax.legend()
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('Z')
#         plt.show()


if __name__ == '__main__':
    control = Controller()
    control.connect(database='tpu', password='2325070307')

    # # control.create_tables()
    # # D:\Projects\mat_to_csv\mat files
    # # control.create_tables()
    # # control.fill_db()
    #
    # # int_1 = control.graphs('isofield', 'mean', [-1, 10000], '4', '115', '0')
    # # int_2 = control.graphs('isofield', 'mean', [-1, 1], '4', '114', '0')
    #
    # control.graphs('isofield', 'mean', [-1, 1], '4', '114', '0')
    # control.graphs('signal', 'mean', [-1, 1], '4', '114', '0')
    new_signal = control.generate_signal('4', '114', '0', 0.14, 0.19)
    control.graphs('signal', 'mean', [-1, 1], '4', '114', '0', new_signal)

    control.disconnect()

    # x1 = [[[0.01, 0.03, 0.05, 0.07, 0.09], [0.01, 0.03, 0.05, 0.07, 0.09], [0.01, 0.03, 0.05, 0.07, 0.09],
    #        [0.01, 0.03, 0.05, 0.07, 0.09], [0.01, 0.03, 0.05, 0.07, 0.09], [0.01, 0.03, 0.05, 0.07, 0.09],
    #        [0.01, 0.03, 0.05, 0.07, 0.09], [0.01, 0.03, 0.05, 0.07, 0.09], [0.01, 0.03, 0.05, 0.07, 0.09],
    #        [0.01, 0.03, 0.05, 0.07, 0.09], [0.01, 0.03, 0.05, 0.07, 0.09], [0.01, 0.03, 0.05, 0.07, 0.09],
    #        [0.01, 0.03, 0.05, 0.07, 0.09], [0.01, 0.03, 0.05, 0.07, 0.09], [0.01, 0.03, 0.05, 0.07, 0.09],
    #        [0.01, 0.03, 0.05, 0.07, 0.09], [0.01, 0.03, 0.05, 0.07, 0.09], [0.01, 0.03, 0.05, 0.07, 0.09],
    #        [0.01, 0.03, 0.05, 0.07, 0.09], [0.01, 0.03, 0.05, 0.07, 0.09], [0.01, 0.03, 0.05, 0.07, 0.09],
    #        [0.01, 0.03, 0.05, 0.07, 0.09], [0.01, 0.03, 0.05, 0.07, 0.09], [0.01, 0.03, 0.05, 0.07, 0.09],
    #        [0.01, 0.03, 0.05, 0.07, 0.09]]]
    #
    # z1 = [[[0.49, 0.49, 0.49, 0.49, 0.49], [0.47, 0.47, 0.47, 0.47, 0.47], [0.45, 0.45, 0.45, 0.45, 0.45],
    #        [0.43, 0.43, 0.43, 0.43, 0.43], [0.41, 0.41, 0.41, 0.41, 0.41], [0.39, 0.39, 0.39, 0.39, 0.39],
    #        [0.37, 0.37, 0.37, 0.37, 0.37], [0.35, 0.35, 0.35, 0.35, 0.35], [0.33, 0.33, 0.33, 0.33, 0.33],
    #        [0.31, 0.31, 0.31, 0.31, 0.31], [0.29, 0.29, 0.29, 0.29, 0.29], [0.27, 0.27, 0.27, 0.27, 0.27],
    #        [0.25, 0.25, 0.25, 0.25, 0.25], [0.23, 0.23, 0.23, 0.23, 0.23], [0.21, 0.21, 0.21, 0.21, 0.21],
    #        [0.19, 0.19, 0.19, 0.19, 0.19], [0.17, 0.17, 0.17, 0.17, 0.17], [0.15, 0.15, 0.15, 0.15, 0.15],
    #        [0.13, 0.13, 0.13, 0.13, 0.13], [0.11, 0.11, 0.11, 0.11, 0.11], [0.09, 0.09, 0.09, 0.09, 0.09],
    #        [0.07, 0.07, 0.07, 0.07, 0.07], [0.05, 0.05, 0.05, 0.05, 0.05], [0.03, 0.03, 0.03, 0.03, 0.03],
    #        [0.01, 0.01, 0.01, 0.01, 0.01]]]
    # results = [[], [], [], []]
    #
    # summa = 0
    # for face in range(4):
    #     for floor in range(len(z1[0])):
    #         results[face].append([])
    #         for point in range(len(z1[0][0])):
    #             raz = float(int_2[face]([[x1[0][floor][point], z1[0][floor][point]]]).round(6))
    #             # - int_2[face]([[x1[0][floor][point], z1[0][floor][point]]])).round(6))
    #             results[face][-1].append(raz)
    #             summa += raz
    # print(results)
    # print('-' * 100)
    # print(results[0])
    # print('-' * 100)
    # print(results[0][0])
    # kolvo = len(z1) * len(z1[0]) * len(z1[0][0])
    # print(kolvo)
    # print(summa / kolvo)
    #
    # raz = [[[0.012985, 0.021689, 0.030928, 0.015613, 0.007989],
    #         [0.029837, 0.064221, 0.0746, 0.045179, 0.002383],
    #         [0.004, 0.009379, 0.002773, 0.004145, 0.011406],
    #         [0.008931, 0.028988, 0.010811, 0.023689, 0.026103],
    #         [0.016042, 0.013506, 0.006219, 0.010948, 0.002821],
    #         [0.013292, 0.04492, 0.017899, 0.010498, 0.024153],
    #         [0.040749, 0.005849, 0.010882, 0.010859, 0.003075],
    #         [0.00093, 0.005922, 0.002973, 0.018105, 0.009672],
    #         [0.010659, 0.010113, 0.012509, 0.009385, 0.022286],
    #         [0.008925, 0.022602, 0.018236, 0.002004, 0.006596],
    #         [0.002151, 0.019567, 0.011087, 0.012395, 0.0292],
    #         [0.02188, 0.011856, 1.1e-05, 0.001575, 0.000258],
    #         [0.001278, 0.004889, 0.01109, 0.004003, 0.014839],
    #         [0.003374, 0.018643, 0.003908, 0.016627, 0.003559],
    #         [0.008141, 0.005766, 0.024894, 0.013086, 0.013484],
    #         [0.015688, 0.020573, 0.014327, 0.0217, 0.014939],
    #         [0.017322, 0.017628, 0.008746, 0.022295, 0.018383],
    #         [0.00411, 0.000492, 0.012075, 0.008494, 0.006597],
    #         [0.023615, 0.002532, 0.00956, 0.001744, 9.9e-05],
    #         [0.006217, 0.014506, 0.0055, 0.017359, 0.005252],
    #         [0.010152, 0.005406, 0.011413, 0.000406, 0.004106],
    #         [0.006236, 0.000512, 0.008723, 0.005702, 0.011207],
    #         [0.02969, 0.001977, 0.016554, 0.007973, 0.016708],
    #         [0.005916, 0.014956, 0.042313, 0.02249, 0.005043],
    #         [0.01166, 0.012622, 0.043865, 0.007934, 0.008661]],
    #
    #        [[0.019109, 0.010419, 0.014866, 0.031955, 0.019611],
    #         [0.011652, 0.022895, 0.028488, 0.046803, 0.047116],
    #         [0.032626, 0.02724, 0.011205, 0.034245, 0.059771],
    #         [0.033159, 0.017011, 0.014085, 0.018789, 0.051913],
    #         [0.011166, 0.019798, 0.01547, 0.016744, 0.041734],
    #         [0.005899, 0.026684, 0.010444, 0.000987, 0.034426],
    #         [0.012542, 0.014529, 0.007117, 0.014838, 0.00517],
    #         [0.014939, 0.00546, 0.008365, 0.018397, 0.009632],
    #         [0.020809, 0.031757, 0.007579, 0.005004, 0.021478],
    #         [0.012412, 0.003364, 0.01197, 0.000651, 0.011986],
    #         [0.04123, 0.043815, 0.023134, 0.005638, 0.034601],
    #         [0.048384, 0.054749, 0.030918, 0.006116, 0.010852],
    #         [0.024314, 0.016736, 0.029912, 0.018285, 0.03649],
    #         [0.030565, 0.041661, 0.009417, 0.005348, 0.032475],
    #         [0.04252, 0.023973, 0.007845, 0.006276, 0.021598],
    #         [0.042816, 0.033776, 0.001108, 0.000882, 0.016366],
    #         [0.055784, 0.056867, 0.015891, 0.001426, 0.028245],
    #         [0.033805, 0.043908, 0.005757, 0.037257, 0.027154],
    #         [0.052803, 0.030091, 0.026542, 0.000776, 0.027264],
    #         [0.060779, 0.042161, 0.020615, 0.004719, 0.030828],
    #         [0.058231, 0.05102, 0.010919, 0.002173, 0.004154],
    #         [0.045848, 0.04053, 0.011238, 0.003732, 0.004229],
    #         [0.046527, 0.048465, 0.008911, 0.001987, 0.007211],
    #         [0.018326, 0.035431, 0.008733, 0.008525, 0.00461],
    #         [0.031521, 0.033857, 0.014068, 0.026871, 0.000891]],
    #
    #        [[0.025168, 0.037021, 0.033106, 0.030508, 0.023339],
    #         [0.021914, 0.033125, 0.040128, 0.036142, 0.030761],
    #         [0.016457, 0.028832, 0.022847, 0.002724, 0.031534],
    #         [0.014356, 0.007555, 0.028179, 0.009317, 0.034554],
    #         [0.020767, 0.017291, 0.006886, 0.018724, 0.027562],
    #         [0.023075, 0.00348, 0.009508, 0.001281, 0.001448],
    #         [0.016589, 0.006741, 0.010798, 0.001966, 0.00777],
    #         [0.018382, 0.026284, 0.013933, 0.017384, 0.019963],
    #         [0.014924, 0.004659, 0.017612, 0.002154, 0.002458],
    #         [0.009105, 0.033962, 0.026399, 0.01202, 0.013087],
    #         [0.014759, 0.005174, 0.020332, 0.015161, 0.016773],
    #         [0.011013, 0.023633, 0.022983, 0.009244, 0.025064],
    #         [0.016869, 0.018613, 0.002289, 0.004488, 0.041162],
    #         [0.025167, 0.040701, 0.022693, 0.01245, 0.020923],
    #         [0.021428, 0.038326, 0.014031, 0.02601, 0.047636],
    #         [0.002214, 0.015626, 0.005794, 0.010979, 0.026588],
    #         [0.020104, 0.014285, 0.02287, 0.013118, 0.002572],
    #         [0.024634, 0.003305, 0.000237, 0.01358, 0.011702],
    #         [0.014356, 0.019566, 0.029377, 0.012596, 0.002818],
    #         [0.022283, 0.006531, 6e-06, 0.015237, 0.030511],
    #         [0.009881, 0.012029, 0.017072, 0.015751, 0.020598],
    #         [0.002043, 0.014656, 0.001598, 0.027415, 0.007707],
    #         [0.017401, 0.01526, 0.010438, 0.007942, 0.010875],
    #         [0.01791, 0.01015, 0.007823, 0.000507, 0.005156],
    #         [0.013826, 0.034912, 0.020092, 0.005085, 0.012534]],
    #
    #        [[0.040652, 0.034204, 0.015386, 0.016125, 0.024413],
    #         [0.045013, 0.035056, 0.011063, 0.028615, 0.036414],
    #         [0.056008, 0.04378, 0.008576, 0.047351, 0.044547],
    #         [0.051688, 0.033312, 0.018262, 0.033224, 0.032976],
    #         [0.038364, 0.024093, 0.006013, 0.035219, 0.025477],
    #         [0.031991, 0.016389, 0.015692, 0.019989, 0.013523],
    #         [0.031635, 0.013711, 0.012925, 0.028674, 0.017596],
    #         [0.010648, 0.015571, 0.01709, 0.038792, 0.039085],
    #         [0.045856, 0.0032, 0.000223, 0.024066, 0.023641],
    #         [0.005328, 0.001587, 0.010382, 0.021328, 0.056474],
    #         [0.016216, 0.010766, 0.023111, 0.032888, 0.044366],
    #         [0.004735, 0.01924, 0.008666, 0.033477, 0.036722],
    #         [0.033325, 0.00229, 0.003405, 0.030672, 0.037103],
    #         [0.027986, 0.001124, 0.003445, 0.029731, 0.052885],
    #         [0.037968, 0.002879, 0.020565, 0.046322, 0.051617],
    #         [0.024315, 0.01698, 0.017956, 0.032586, 0.098741],
    #         [0.001451, 0.00235, 0.010163, 0.032576, 0.044872],
    #         [0.018764, 0.00819, 0.021407, 0.006691, 0.043525],
    #         [0.002892, 0.022559, 0.003178, 0.028771, 0.015453],
    #         [0.016274, 0.012938, 0.009426, 0.020524, 0.012163],
    #         [0.016635, 0.005929, 0.013814, 0.029019, 0.033819],
    #         [0.032059, 0.002614, 0.007291, 0.047901, 0.009421],
    #         [0.002566, 0.009827, 0.004685, 0.043668, 0.069009],
    #         [0.006852, 0.000197, 0.005809, 0.019306, 0.038916],
    #         [0.012327, 0.007611, 0.012446, 0.035302, 0.026103]]]
    # summa = []
    # for i in range(len(raz)):
    #     summa_temp = 0
    #     k_temp = 0
    #     for j in range(len(raz[0])):
    #         k_temp += 5
    #         summa_temp += sum(raz[i][j])
    #     summa.append(summa_temp / k_temp)
    # print(sum(summa) / 4)
    #
    # val114ist = [[[0.478953, 0.563174, 0.562375, 0.553837, 0.477864],
    #               [0.573688, 0.755789, 0.77738, 0.744574, 0.563543],
    #               [0.599643, 0.834139, 0.871788, 0.822321, 0.577394],
    #               [0.581255, 0.828394, 0.867975, 0.8223, 0.566723],
    #               [0.563737, 0.837411, 0.874053, 0.826698, 0.56634],
    #               [0.558822, 0.845538, 0.861728, 0.805987, 0.544861],
    #               [0.54956, 0.80601, 0.854175, 0.787114, 0.517957],
    #               [0.523767, 0.780861, 0.833727, 0.768977, 0.498643],
    #               [0.498621, 0.767799, 0.808894, 0.749507, 0.479497],
    #               [0.484771, 0.742577, 0.785739, 0.735761, 0.46488],
    #               [0.470764, 0.717431, 0.755051, 0.714418, 0.464951],
    #               [0.437484, 0.690127, 0.732999, 0.685403, 0.432889],
    #               [0.418437, 0.662884, 0.710825, 0.657823, 0.402413],
    #               [0.410686, 0.63987, 0.683807, 0.632696, 0.38359],
    #               [0.389181, 0.615376, 0.652469, 0.603885, 0.362657],
    #               [0.359632, 0.581971, 0.620914, 0.57434, 0.341931],
    #               [0.336351, 0.554064, 0.590094, 0.545669, 0.3188],
    #               [0.311224, 0.524366, 0.561239, 0.514678, 0.297778],
    #               [0.285842, 0.492666, 0.530086, 0.483878, 0.272814],
    #               [0.265146, 0.462516, 0.493196, 0.453933, 0.241362],
    #               [0.239668, 0.429619, 0.456479, 0.418586, 0.216508],
    #               [0.204622, 0.395532, 0.427784, 0.386298, 0.190783],
    #               [0.16865, 0.361019, 0.401048, 0.35382, 0.154857],
    #               [0.158715, 0.37144, 0.418232, 0.364846, 0.142295],
    #               [0.180642, 0.427973, 0.478061, 0.422456, 0.169838]],
    #
    #              [[-1.034813, -1.064716, -0.94648, -0.770331, -0.626358],
    #               [-0.998306, -1.044148, -0.974967, -0.808477, -0.67485],
    #               [-0.949735, -1.000967, -0.978706, -0.840215, -0.721837],
    #               [-0.912406, -0.966283, -0.96695, -0.853486, -0.75836],
    #               [-0.890217, -0.946947, -0.95233, -0.855634, -0.784905],
    #               [-0.88491, -0.933261, -0.937853, -0.855282, -0.795573],
    #               [-0.885501, -0.924945, -0.929802, -0.848714, -0.79979],
    #               [-0.882995, -0.921389, -0.924037, -0.848815, -0.795466],
    #               [-0.880064, -0.921205, -0.922464, -0.849227, -0.785849],
    #               [-0.88088, -0.921532, -0.92376, -0.841347, -0.775066],
    #               [-0.884171, -0.920398, -0.919684, -0.831909, -0.761342],
    #               [-0.888805, -0.926107, -0.910931, -0.821207, -0.745007],
    #               [-0.889076, -0.926359, -0.905878, -0.809439, -0.731161],
    #               [-0.88806, -0.922486, -0.900596, -0.797137, -0.715068],
    #               [-0.892589, -0.922331, -0.889908, -0.781516, -0.69442],
    #               [-0.898897, -0.924299, -0.877051, -0.760363, -0.681236],
    #               [-0.903601, -0.920759, -0.862229, -0.750807, -0.670673],
    #               [-0.90206, -0.914331, -0.848649, -0.736361, -0.654951],
    #               [-0.895458, -0.904834, -0.835512, -0.716585, -0.635012],
    #               [-0.88708, -0.889479, -0.819151, -0.697857, -0.616705],
    #               [-0.878442, -0.874423, -0.795418, -0.680076, -0.596223],
    #               [-0.870989, -0.86629, -0.783131, -0.661489, -0.567606],
    #               [-0.861065, -0.870192, -0.769014, -0.628433, -0.532911],
    #               [-0.850769, -0.859266, -0.716384, -0.564995, -0.486094],
    #               [-0.82456, -0.806993, -0.629828, -0.492055, -0.429934]],
    #
    #              [[-0.588491, -0.57693, -0.571299, -0.572143, -0.581217],
    #               [-0.597514, -0.587651, -0.588642, -0.590631, -0.593387],
    #               [-0.602966, -0.589699, -0.587471, -0.589479, -0.600861],
    #               [-0.601459, -0.580815, -0.574876, -0.577063, -0.602658],
    #               [-0.593934, -0.565207, -0.562931, -0.566541, -0.597877],
    #               [-0.587017, -0.553128, -0.542535, -0.553316, -0.58778],
    #               [-0.5784, -0.535292, -0.52769, -0.539569, -0.575992],
    #               [-0.566738, -0.521991, -0.516623, -0.526545, -0.563638],
    #               [-0.554906, -0.513036, -0.506077, -0.517438, -0.555662],
    #               [-0.546583, -0.507336, -0.496219, -0.511727, -0.549416],
    #               [-0.54182, -0.501023, -0.494483, -0.501358, -0.538177],
    #               [-0.531918, -0.482089, -0.49198, -0.500398, -0.534505],
    #               [-0.526231, -0.4826, -0.486563, -0.498911, -0.533697],
    #               [-0.524865, -0.493598, -0.485287, -0.496062, -0.531906],
    #               [-0.520806, -0.492236, -0.487738, -0.497176, -0.531738],
    #               [-0.52018, -0.491927, -0.485833, -0.500922, -0.53179],
    #               [-0.524346, -0.498819, -0.491914, -0.502911, -0.531336],
    #               [-0.52483, -0.505794, -0.494648, -0.509786, -0.535152],
    #               [-0.518106, -0.505779, -0.492529, -0.512462, -0.535285],
    #               [-0.513113, -0.504144, -0.494668, -0.508231, -0.527028],
    #               [-0.515873, -0.504489, -0.497607, -0.508924, -0.519085],
    #               [-0.510769, -0.497404, -0.491646, -0.505098, -0.511362],
    #               [-0.501356, -0.496105, -0.49044, -0.503646, -0.503293],
    #               [-0.490261, -0.489935, -0.481445, -0.497266, -0.494584],
    #               [-0.472909, -0.465702, -0.454437, -0.475951, -0.480117]],
    #
    #              [[-0.620469, -0.775596, -0.964196, -1.085805, -1.065773],
    #               [-0.663822, -0.803924, -0.989759, -1.074648, -1.024249],
    #               [-0.711632, -0.836585, -0.994838, -1.038026, -0.975769],
    #               [-0.748334, -0.856663, -0.985633, -1.003496, -0.943651],
    #               [-0.770436, -0.863591, -0.974676, -0.981567, -0.925677],
    #               [-0.779434, -0.869817, -0.968203, -0.966615, -0.916539],
    #               [-0.781334, -0.872598, -0.965566, -0.964487, -0.918014],
    #               [-0.781942, -0.872107, -0.963132, -0.963562, -0.920333],
    #               [-0.773831, -0.865026, -0.960645, -0.962686, -0.920279],
    #               [-0.760331, -0.853908, -0.957954, -0.965304, -0.923222],
    #               [-0.756433, -0.845587, -0.950404, -0.966485, -0.930526],
    #               [-0.736033, -0.834272, -0.943016, -0.971877, -0.931779],
    #               [-0.717003, -0.816917, -0.930503, -0.971699, -0.936151],
    #               [-0.702453, -0.801075, -0.918383, -0.969594, -0.94337],
    #               [-0.685404, -0.789352, -0.913684, -0.971179, -0.949065],
    #               [-0.671545, -0.774823, -0.907738, -0.970688, -0.953192],
    #               [-0.654376, -0.757819, -0.891115, -0.966428, -0.950097],
    #               [-0.645715, -0.743091, -0.870833, -0.953575, -0.947225],
    #               [-0.630574, -0.726123, -0.851017, -0.938097, -0.931903],
    #               [-0.598708, -0.702478, -0.831389, -0.929416, -0.91098],
    #               [-0.571285, -0.675336, -0.812349, -0.919083, -0.918341],
    #               [-0.542526, -0.642998, -0.789364, -0.914455, -0.910583],
    #               [-0.506162, -0.606515, -0.769596, -0.908789, -0.913639],
    #               [-0.459963, -0.544663, -0.710319, -0.885343, -0.917207],
    #               [-0.407814, -0.468095, -0.609024, -0.825034, -0.881209]]]

# paths = control.get_paths()
# import time
# users = dict()
# for i in range(1, 80):
#     users[f'user_{i}'] = Controller()
#     users[f'user_{i}'].connect(database='', password='')
#
#
#
# for i in range(1, 80):
#     users[f'user_{i}'].disconnect()
