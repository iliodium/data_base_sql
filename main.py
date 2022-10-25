import psycopg2
from psycopg2 import Error
import os
import scipy.io as sio
import uuid


class Controller:
    """
    dasdasd
    sadsadsa
    d
    sa
    das
    dsad
    """
    __connection = None
    # __path_database = None
    __path_database = 'D:\Projects\mat_to_csv\mat files'

    def connect(self, database = 'tpu', password = '2325070307', user = "postgres", host = "127.0.0.1", port = "5432"):
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
            print("Проверьте правильность веденных данных")
        except Exception as error:
            print(f"Ошибка - {error}")

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
        count_sensors = sensor_number[-1]
        return (alpha,
                model_name,
                breadth,
                depth,
                height,
                frequency,
                period,
                speed,
                x,
                z,
                sensor_number,
                face_number,
                angle,
                pressure_coefficients,
                count_sensors)

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

    def fill_models_alpha(self, alpha, model_name, angle, pressure_coefficients):
        flag = 0
        try:
            name = f"T{model_name}_{alpha}_{angle:03d}_1.mat"
            command = f"""
                       select model_id 
                       from experiments_alpha_{alpha}
                       where model_name = {model_name}
                   """
            self.cursor.execute(command)
            model_id = self.cursor.fetchall()[0][0]

            command = f"""
                       select model_id 
                       from models_alpha_{alpha}
                       where model_id = {model_id} and angle = {angle}
                               """
            self.cursor.execute(command)

            if self.cursor.fetchall():
                print(f'{name} была ранее добавлена в models_alpha_{alpha}')
                flag = 1

            if flag == 0:
                norm_pressure_coefficients = ''
                for step_time in pressure_coefficients:
                    norm_pressure_coefficients += f'{step_time},'.replace('[', '{').replace(']', '}')
                norm_pressure_coefficients = norm_pressure_coefficients.rstrip(',')
                try:
                    command = f"""
                               insert into models_alpha_{alpha} (angle, pressure_coefficients)
                               values
                               ({angle}, array[{norm_pressure_coefficients}])
                           """
                    self.cursor.execute(command)
                    self.__connection.commit()
                    print(f'{name} добавлена в models_alpha_{alpha}')
                except (Exception, Error) as error:
                    print(f"Ошибка при работе с PostgreSQL, файл {model_name}", error)
        except (Exception, Error) as error:
            print(f"Ошибка при работе с PostgreSQL, файл {model_name}", error)

    def fill_experiments_alpha(self,
                               alpha,
                               angle,
                               model_name,
                               breadth,
                               depth,
                               height,
                               frequency,
                               period,
                               speed,
                               x_old,
                               z,
                               sensor_number,
                               face_number,
                               count_sensors):
        flag = 0  # флаг для проверки наличия записи в бд
        x, y = self.converter_coordinates(x_old, depth, breadth, face_number, count_sensors)
        name = f"T{model_name}_{alpha}_{angle:03d}_1.mat"
        try:
            command = f"""
                       select * from experiments_alpha_{alpha}
                   """
            self.cursor.execute(command)
            table = self.cursor.fetchall()
            for row in table:
                if row[1] == model_name:
                    print(f'{name} была ранее добавлена в experiments_alpha_{alpha}')
                    flag = 1
                    break
            if flag == 0:
                try:
                    command = f"""
                               insert into experiments_alpha_{alpha} (model_name,
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
                               values
                               (
                               {model_name},
                               {breadth},
                               {depth},
                               {height},
                               {frequency},
                               {period},
                               {speed},
                               array{x},
                               array{y},
                               array{z},
                               array{sensor_number},
                               array{face_number}
                               )
                           """
                    self.cursor.execute(command)
                    self.__connection.commit()
                    print(f'{name} добавлена в experiments_alpha_{alpha}')
                except (Exception, Error) as error:
                    print(f"Ошибка при работе с PostgreSQL, файл {model_name}", error)
        except (Exception, Error) as error:
            print(f"Ошибка при работе с PostgreSQL, файл {model_name}", error)

    def fill_db(self, path = None):
        """
        Метод заполняет таблицу experiments_alpha_<alpha> и models_alpha_<alpha> данными из .mat файла.

        Если параметры не переданы, то в таблицы добавляются все данные из директории.
        """

        if path:
            (alpha,
             model_name,
             breadth,
             depth,
             height,
             frequency,
             period,
             speed,
             x_old,
             z,
             sensor_number,
             face_number,
             angle,
             pressure_coefficients,
             count_sensors) = self.read_mat(path)
            self.fill_experiments_alpha(alpha,
                                        angle,
                                        model_name,
                                        breadth,
                                        depth,
                                        height,
                                        frequency,
                                        period,
                                        speed,
                                        x_old,
                                        z,
                                        sensor_number,
                                        face_number,
                                        count_sensors)

            self.fill_models_alpha(alpha, model_name, angle, pressure_coefficients)

        else:
            self.check_path()
            for alpha in os.listdir(self.__path_database):
                for files in os.listdir(f'{self.__path_database}\\{alpha}'):
                    for file in os.listdir(f"{self.__path_database}\\{alpha}\\{files}"):
                        self.fill_db(f"{self.__path_database}\\{alpha}\\{files}\\{file}")

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

    def create_tables(self, alpha = None):
        """
        Метод создаёт таблицы с заданным параметром альфа.

        Если альфа не передан создаются таблицы с параметрами из директории с файлами.
        """
        if alpha:
            # создается таблица с экспериментами
            try:
                command = f'''
                    create table experiments_alpha_{alpha}
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
                    );'''
                self.cursor.execute(command)
                self.__connection.commit()
                print(f"Создана таблица с экспериментами с параметром {alpha}")
            except Exception as error:
                print(f"Ошибка -> {error}")
                self.__connection.commit()

            # создается таблица с моделями
            try:
                command = f'''
                    create table models_alpha_{alpha}
                    (
                        model_id serial not null,
                        angle smallint not null,
                        pressure_coefficients smallint[][] not null,
                        
                        constraint FK{str(uuid.uuid4()).replace('-', '')} foreign key (model_id) references experiments_alpha_{alpha}(model_id)
                    );'''
                self.cursor.execute(command)
                self.__connection.commit()
                print(f"Создана таблица с моделями с параметром {alpha}")
            except Exception as error:
                print(f"Ошибка -> {error}")
                self.__connection.commit()
        else:
            self.check_path()
            for alpha in os.listdir(self.__path_database):
                alp = alpha[-1]
                control.create_tables(alp)


if __name__ == '__main__':
    control = Controller()
    # control.read_mat('D:\Projects\mat_to_csv\mat files\Alpha 4\T111_4\T111_4_000_1.mat')
    control.connect()
    control.create_tables()
    control.fill_db()
    control.disconnect()
