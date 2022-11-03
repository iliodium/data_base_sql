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

        x, z = coordinates
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
            coords = [[i1, j1] for i1, j1 in zip(x_old, z_old)]  # Старые координаты
            # Интерполятор полученный на основе имеющихся данных
            interpolator = Artist.interpolator(coords, data_old)
            integral_func.append(interpolator)
            # Получаем данные для несуществующих датчиков
            data_new = [float(interpolator([[X, Y]])) for X, Y in zip(x_new, z_new)]

            triang = mtri.Triangulation(x_new, z_new)
            refiner = mtri.UniformTriRefiner(triang)
            grid, value = refiner.refine_field(data_new, subdiv=4)

            data_colorbar = graph[i].tricontourf(grid, value, cmap=cmap, levels=levels, extend='both')
            aq = graph[i].tricontour(grid, value, linewidths=1, linestyles='solid', colors='black', levels=levels)

            graph[i].clabel(aq, fontsize=13)
            graph[i].set_ylim([0, height])
            graph[i].set_aspect('equal')
            if i in [0, 2]:
                graph[i].set_xlim([0, breadth])
                graph[i].set_xticks(ticks=np.arange(0, breadth + 0.1, 0.1))
            else:
                graph[i].set_xlim([0, depth])
                graph[i].set_xticks(ticks=np.arange(0, depth + 0.1, 0.1))

        fig.colorbar(data_colorbar, ax=graph, location='bottom', cmap=cmap, ticks=levels)
        # plt.show()
        plt.close()

        if integral[0] == -1:
            return None
        elif integral[0] == 0:
            count_zone = count_row
        else:
            count_zone = integral[0]
        ran = random.uniform
        face1, face2, face3, face4 = [], [], [], []
        model = face1, face2, face3, face4
        count_point = integral[1]
        step_floor = height / count_row
        for i in range(4):
            top = step_floor
            down = 0
            for _ in range(count_zone):
                floor = []
                if i in [0, 2]:
                    for _ in range(count_point):
                        floor.append(integral_func[i]([[ran(0, breadth), ran(down, top)]]))
                else:
                    for _ in range(count_point):
                        floor.append(integral_func[i]([[ran(0, depth), ran(down, top)]]))
                down = top
                top += step_floor
                # Обычный метод Монте-Синякина
                if i in [0, 2]:
                    model[i].append(sum(floor) * breadth / count_point)
                else:
                    model[i].append(sum(floor) * depth / count_point)
        print(face1)
        print(face2)
        print(face3)
        print(face4)

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
    __extrapolatedAnglesInfoList = {}  # ключ вида T<model_name>_<alpha>_<angle>, значения (коэффициенты, x, y, z)

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

    def graphs(self, graphic, mode, integtal, alpha, model_name, angle):
        """Метод отвечает за вызов графика из класса Artist"""
        angle = int(angle) % 360
        if angle % 5 != 0:
            print('Углы должны быть кратны 5')
            return None
        # modes_graphs = {
        #     'isofield_min': Artist.isofield_min,
        #     'isofield_mean': Artist.isofield_mean,
        #     'isofield_max': Artist.isofield_max,
        #     'isofield_std': Artist.isofield_std,
        #     'signal': Artist.signal,
        #     'spectrum': Artist.spectrum,
        # }
        pressure_coefficients = np.array(self.get_pressure_coefficients(alpha, model_name, angle)) / 1000
        # pressure_coefficients1 = np.array(self.get_pressure_coefficients('4', '113', '0')) / 1000
        try:
            coordinates = self.__extrapolatedAnglesInfoList[f'T{model_name}_{alpha}_{angle:03d}'][1:]
        except:
            coordinates = self.get_coordinates(alpha, model_name)
        # modes_graphs[mode](pressure_coefficients, pressure_coefficients1, alpha, model_name, angle)
        Artist.isofield(mode, pressure_coefficients, coordinates, integtal, alpha, model_name, angle)


if __name__ == '__main__':
    Artist.check()
    # control = Controller()
    # control.connect(database='tpu', password='2325070307')

    # control.create_tables()
    # D:\Projects\mat_to_csv\mat files
    #
    # control.generate_not_exists_case('4', '111', '65')
    # control.graphs('isofield_mean', '4', '111', '0')
    # control.graphs('isofield_mean', '4', '112', '0')
    # control.graphs('isofield_mean', '4', '113', '0')
    # control.graphs('isofield_mean', '4', '114', '0')
    # control.get_coordinates(4, 313)
    # control.graphs('isofield_mean', '4', '111', '20')
    # control.create_tables()
    # control.fill_db()

    # control.graphs('isofield', 'mean', [0, 10000], '4', '112', '30')
    # control.disconnect()

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
