# Система управления БД
Программа используя `Python` взаимодействует с базой данных `PostgreSQL`.

## Структура

0. [Взаимодействие с PostgreSQL](#Взаимодействие с PostgreSQL)
1. [Структура БД](#Структура БД)
2. [Обработка mat файлов](#Обработка mat файлов)
3. [Список задач](#Список задач)

## Взаимодействие с PostgreSQL
В `PostgreSQL` хранятся обработанные данные, благодаря чему удалось снизить размер БД на 41%, данные изначально хранились как набор
`mat` файлов.

## Структура БД
База данных представляет собой 2 пары взаимосвязанных таблиц
с заданным параметром `альфа`.
В дальнейшем будет добавлена 3 пара, полученная путем интерполяции 
данных от имеющихся двух пар.
```PostgreSQL
create table experiments_alpha_<ALPHA>
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
);
create table models_alpha_<ALPHA>
(
	model_id serial not null,
	angle smallint not null,
	pressure_coefficients smallint[][] not null,
	
	constraint FK_<UUID> foreign key (model_id) references experiments_alpha_4(model_id)
);
```

## Обработка mat файлов
Файлы с расширением `mat` обрабатываются с помощью `Python` и библиотеки
`NumPy`. Для достижения наименьшего объема каждое 
значение из матрицы  `pressure_coefficients` домнажалось на 1000 и математически
округлялось в верх, дробная часть отбрасывалась. Полученная точность
достаточна.

## Список задач
- Оптимизировать PostgreSQL запросы.
- Реализовать модуль для создания и отображения графиков.