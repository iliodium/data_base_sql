import os
import uuid
import psycopg2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from psycopg2 import Error


class Artist:
    @staticmethod
    def izofields_min(parameters):
        pass


class Controller:
    """
    Через класс происходит управление базой данных:
        -создание таблиц
        -заполнение таблиц
        -генерация несуществующих вариантов
    """
    __connection = None
    __path_database = None
    __extrapolatedAnglesInfoList = {}  # ключ вида T<model_name>_<alpha>_<angle>

    # __path_database = 'D:\Projects\mat_to_csv\mat files'

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
        x_old = [float('%.5f' % i) for i in mat_file["Location_of_measured_points"][0]]
        z = [float('%.5f' % i) for i in mat_file["Location_of_measured_points"][1]]
        sensor_number = [int(i) for i in mat_file["Location_of_measured_points"][2]]
        face_number = [int(i) for i in mat_file["Location_of_measured_points"][3]]
        angle = int(mat_file['Wind_direction_angle'][0][0])
        pressure_coefficients = mat_file["Wind_pressure_coefficients"] * 1000
        pressure_coefficients = pressure_coefficients.round(0)
        pressure_coefficients = pressure_coefficients.astype('int32')
        pressure_coefficients = pressure_coefficients.tolist()
        count_sensors = sensor_number[-1]
        return {'alpha': alpha,
                'model_name': model_name,
                'breadth': breadth,
                'depth': depth,
                'height': height,
                'frequency': frequency,
                'period': period,
                'speed': speed,
                'x_old': x_old,
                'z': z,
                'sensor_number': sensor_number,
                'face_number': face_number,
                'angle': angle,
                'pressure_coefficients': pressure_coefficients,
                'count_sensors': count_sensors}

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
        x_old = parameters['x_old']
        z = parameters['z']
        sensor_number = parameters['sensor_number']
        face_number = parameters['face_number']
        count_sensors = parameters['count_sensors']
        flag = 0  # флаг для проверки наличия записи в бд
        x, y = self.converter_coordinates(x_old, depth, breadth, face_number, count_sensors)
        name = f"T{model_name}_{alpha}_{angle:03d}_1.mat"
        try:
            if alpha == '6':
                self.cursor.execute("""
                           select * from experiments_alpha_6
                       """)
            elif alpha == '4':
                self.cursor.execute("""
                           select * from experiments_alpha_4
                       """)
            table = self.cursor.fetchall()
            for row in table:
                if row[1] == model_name:
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
                                                                    y_coordinates,
                                                                    z_coordinates,
                                                                    sensor_number,
                                                                    face_number)
                                   values((%s),(%s),(%s),(%s),(%s),(%s),(%s),(%s),(%s),(%s),(%s),(%s))
                               """, (
                            model_name, breadth, depth, height, frequency, period, speed, x, y, z, sensor_number,
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
                                                                    y_coordinates,
                                                                    z_coordinates,
                                                                    sensor_number,
                                                                    face_number)
                                   values((%s),(%s),(%s),(%s),(%s),(%s),(%s),(%s),(%s),(%s),(%s),(%s))
                               """, (
                            model_name, breadth, depth, height, frequency, period, speed, x, y, z, sensor_number,
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
                    y_coordinates real[] not null,
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
                    y_coordinates real[] not null,
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
                    
                    constraint FK_{str(uuid.uuid4()).replace('-', '')} foreign key (model_id) references experiments_alpha_4(model_id)
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

                    constraint FK_{str(uuid.uuid4()).replace('-', '')} foreign key (model_id) references experiments_alpha_6(model_id)
                )""")
            self.__connection.commit()
            print(f"Создана таблица с моделями с параметром 6")

        except Exception as error:
            print(f"Ошибка -> {error}")
            self.__connection.commit()

    def generate_not_exists_case(self, alpha, model_name, angle):
        print(f'Генерация модели {model_name} с параметром {alpha} и углом {angle}')
        if model_name[0] == model_name[1]:  # в основании квадрат
            if any([45 < angle < 90, 135 < angle < 180, 225 < angle < 270, 315 < angle < 360]):
                self.__extrapolatedAnglesInfoList[f'T{model_name}_{alpha}_{angle:03d}'] = self.generation('rev',
                                                                                                          alpha,
                                                                                                          model_name,
                                                                                                          angle)
            else:
                self.__extrapolatedAnglesInfoList[f'T{model_name}_{alpha}_{angle:03d}'] = self.generation('for',
                                                                                                          alpha,
                                                                                                          model_name,
                                                                                                          angle)
        else:  # в основании прямоугольник
            if any([90 < angle < 180, 270 < angle < 360]):
                self.__extrapolatedAnglesInfoList[f'T{model_name}_{alpha}_{angle:03d}'] = self.generation('rev',
                                                                                                          alpha,
                                                                                                          model_name,
                                                                                                          angle)
            else:
                self.__extrapolatedAnglesInfoList[f'T{model_name}_{alpha}_{angle:03d}'] = self.generation('for',
                                                                                                          alpha,
                                                                                                          model_name,
                                                                                                          angle)

    def generation(self, type_gen, alpha, model_name, angle):
        if model_name[0] == model_name[1]:
            types_reverse = {
                45 < angle < 90: (1, 0, 3, 2),
                135 < angle < 180: (2, 1, 0, 3),
                225 < angle < 270: (3, 2, 1, 0),
                315 < angle < 360: (0, 3, 1, 2)
            }
            types_forward = {
                0 <= angle <= 45: (0, 1, 2, 3),
                90 <= angle <= 135: (3, 0, 1, 2),
                180 <= angle <= 225: (2, 3, 0, 1),
                270 <= angle <= 315: (1, 2, 3, 0)
            }  # параметры для перестановки данных
            if type_gen == 'rev':
                view_data = types_reverse[True]
                angle_parent = 90 * (angle // 90 + 1) - angle
            else:
                view_data = types_forward[True]
                angle_parent = angle % 90
            return self.norm_pressure_coefficients(alpha, model_name, angle_parent, view_data, type_gen)
        else:
            types_reverse = {
                90 < angle < 180: (2, 1, 0, 3),
                270 < angle < 360: (0, 3, 2, 1),
            }
            types_forward = {
                0 <= angle <= 90: (0, 1, 2, 3),
                180 <= angle <= 270: (2, 3, 0, 1)
            }  # параметры для перестановки данных
            if type_gen == 'rev':
                view_data = types_reverse[True]
                angle_parent = 90 * (angle // 90 + 1) - angle
            else:
                view_data = types_forward[True]
                angle_parent = angle % 90
            return self.norm_pressure_coefficients(alpha, model_name, angle_parent, view_data, type_gen)

    def norm_pressure_coefficients(self, alpha, model_name, angle, view_data = (0, 1, 2, 3), type_gen = 'for'):
        """Нормирует коэффициенты давления"""
        breadth, depth, count_sens, pressure_coefficients = self.get_parent_pressure_coefficients(alpha,
                                                                                                  model_name,
                                                                                                  angle)

        count_sens_middle_face = int(breadth * 5 * 10)
        count_sens_side_face = int(depth * 5 * 10)
        count_sens_in_row = 2 * (count_sens_middle_face + count_sens_side_face)
        count_row = count_sens // count_sens_in_row
        f1, f2, f3, f4 = view_data
        if type_gen == 'rev':
            for i in range(len(pressure_coefficients)):
                pressure_coefficients[i].reverse()
                pressure_coefficients[i] = np.array(pressure_coefficients[i]).reshape(
                    (count_row, 4, count_sens_middle_face)).swapaxes(0, 1) / 1000
                pressure_coefficients[i][0], pressure_coefficients[i][1], pressure_coefficients[i][2], \
                pressure_coefficients[i][3] = \
                    np.copy(pressure_coefficients[i][f1]), np.copy(pressure_coefficients[i][f2]), np.copy(
                        pressure_coefficients[i][f3]), np.copy(pressure_coefficients[i][f4])
        else:
            for i in range(len(pressure_coefficients)):
                pressure_coefficients[i] = np.array(pressure_coefficients[i]).reshape(
                    (count_row, 4, count_sens_middle_face)).swapaxes(0, 1) / 1000
                pressure_coefficients[i][0], pressure_coefficients[i][1], pressure_coefficients[i][2], \
                pressure_coefficients[i][3] = np.copy(pressure_coefficients[i][f1]), np.copy(
                    pressure_coefficients[i][f2]), \
                                              np.copy(pressure_coefficients[i][f3]), np.copy(
                    pressure_coefficients[i][f4])
        return pressure_coefficients

    def get_parent_pressure_coefficients(self, alpha, model_name, angle):
        """Возвращает breadth, depth, count_sens, pressure_coefficients из таблицы models_alpha_<alpha>"""
        if alpha == '4' or alpha == 4:
            self.cursor.execute("""
                select breadth, depth, model_id
                from experiments_alpha_4
                where model_name = (%s)
            """, (model_name,))
        elif alpha == '6' or alpha == 6:
            self.cursor.execute("""
                select breadth, depth, model_id
                from experiments_alpha_6
                where model_name = (%s)
            """, (model_name,))
        self.__connection.commit()
        breadth, depth, model_id = self.cursor.fetchall()[0]
        if alpha == '4' or alpha == 4:
            self.cursor.execute("""
                select pressure_coefficients
                from models_alpha_4
                where model_id = (%s) and angle = (%s)
            """, (model_id, angle))
        elif alpha == '6' or alpha == 6:
            self.cursor.execute("""
                select pressure_coefficients
                from models_alpha_6
                where model_id = (%s) and angle = (%s)
            """, (model_id, angle))
        self.__connection.commit()
        pressure_coefficients = self.cursor.fetchall()[0][0]
        count_sens = len(pressure_coefficients[0])

        return breadth, depth, count_sens, pressure_coefficients

    def get_pressure_coefficients(self, alpha, model_name, angle):
        """Возвращает нормированные коэффициенты давления"""
        if alpha == '4' or alpha == 4:
            self.cursor.execute("""
                select model_id
                from models_alpha_4
                where model_id = (
                select model_id
                from experiments_alpha_4
                where model_name = (%s)
                ) and angle = (%s)
            """, (model_name, angle))

        elif alpha == '6' or alpha == 6:
            self.cursor.execute("""
                select model_id
                from models_alpha_6
                where model_id = (
                select model_id
                from experiments_alpha_6
                where model_name = (%s)
                ) and angle = (%s)
            """, (model_name, angle))

        self.__connection.commit()

        model_id = self.cursor.fetchall()
        if len(model_id) == 0:
            self.generate_not_exists_case(alpha, model_name, angle)
            return self.__extrapolatedAnglesInfoList[f'T{model_name}_{alpha}_{angle:03d}']

        return self.norm_pressure_coefficients(alpha, model_name, angle)

    def graphs(self, type, alpha, model_name, angle):
        angle = int(angle) % 360
        if angle % 5 != 0:
            print('Углы должны быть кратны 5')
            return None
        types_graphs = {
            'izofields_min': Artist.izofields_min
        }
        pressure_coefficients = self.get_pressure_coefficients(alpha, model_name, angle)

        pressure_coefficients_mean = np.mean(pressure_coefficients, axis=0)

        min_value, max_value = np.min(pressure_coefficients_mean), np.max(pressure_coefficients_mean)

        levels = np.arange(min_value - 0.1, max_value + 0.1, 0.1)
        levels = [float('%.1f' % i) for i in levels]

        pressure_coefficients_mean[0] = np.flip(pressure_coefficients_mean[0], 0)
        pressure_coefficients_mean[1] = np.flip(pressure_coefficients_mean[1], 0)
        pressure_coefficients_mean[2] = np.flip(pressure_coefficients_mean[2], 0)
        pressure_coefficients_mean[3] = np.flip(pressure_coefficients_mean[3], 0)
        data_for_drawing = (
            pressure_coefficients_mean[0],
            pressure_coefficients_mean[1],
            pressure_coefficients_mean[2],
            pressure_coefficients_mean[3])
        type_fig = 'cude'  ####
        heights_arr = []
        widths_arr = []

        for i in data_for_drawing:
            heights_arr.append(len(i))
            widths_arr.append(len(i[0]))
        breadth, depth, height = int(model_name[0]) / 10, int(model_name[1]) / 10, int(model_name[2]) / 10
        fig, graph = plt.subplots(1, 4, figsize=(16, 9), gridspec_kw={'width_ratios': widths_arr})
        fig.suptitle('Mean values', fontsize=20, x=0.7, y=0.95)
        fig.text(0.03, 0.92, f'Model geometrical parameters: H={height}m, B={breadth}m, D={depth}m.\n'
                             f'Wind field parameters: α = 1\\{alpha}, ϴ = {angle}.', fontsize=16)
        for i, j in zip(range(4), data_for_drawing):
            graph[i].set_title(f'Face: {i + 1}')
            contour_data = graph[i].contour(j, levels=levels, linewidths=1, linestyles='solid', colors='black')
            graph[i].clabel(contour_data, fontsize=15)
            data_for_colorbar = graph[i].contourf(j, levels=levels, cmap="jet", extend='max')
            height_arr = heights_arr[i]
            width_arr = widths_arr[i]
            graph[i].set_xlim([0.5, width_arr - 1])
            graph[i].set_ylim([0.5, height_arr - 1])
            graph[i].set_yticks(ticks=np.linspace(0.5, height_arr - 1, int(height * 20 + 1)),
                                labels=map(str, np.round(np.linspace(0, height, int(height * 20 + 1)), 2)))
            if i in [0, 2]:
                graph[i].set_xticks(ticks=np.linspace(0.5, width_arr - 1, int(breadth * 20 + 1)),
                                    labels=map(str, np.round(np.linspace(0, breadth, int(breadth * 20 + 1)), 2)))
            else:
                graph[i].set_xticks(ticks=np.linspace(0.5, width_arr - 1, int(depth * 20 + 1)),
                                    labels=map(str, np.round(np.linspace(0, depth, int(depth * 20 + 1)), 2)))
                graph[i].set_box_aspect(None)
            if type_fig == 'cube':
                graph[i].set_aspect('equal')
        fig.colorbar(data_for_colorbar, ax=graph, location='bottom', cmap="jet", ticks=levels)
        plt.show()


if __name__ == '__main__':
    control = Controller()
    control.connect(database='tpu', password='2325070307')
    # control.create_tables()
    # D:\Projects\mat_to_csv\mat files
    # control.fill_db()
    # control.generate_not_exists_case('4', '111', '65')
    control.graphs('izofields_min', '4', '111', '0')
    control.disconnect()
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
