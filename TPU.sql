select * from alpha_4

select * from models_4

TRUNCATE TABLE alpha_4 RESTART IDENTITY;


drop table models_41;
drop table if exists models_4;



CREATE EXTENSION "uuid-ossp"

SELECT uuid_generate_v1();

insert into experiments_alpha_4(model_name)
values
(111),
(113),
(114),
(115),
(116),
(178)
SHOW max_connections;
SELECT COUNT(*) FROM pg_stat_activity;
select * from experiments_alpha_4
select * from experiments_alpha_6

select * from models_alpha_4
select * from models_alpha_6

drop table if exists experiments_alpha_4;
drop table if exists experiments_alpha_6;

drop table if exists models_alpha_4;
drop table if exists models_alpha_6;
drop table if exists experiments_alpha_4;
drop table if exists experiments_alpha_6;


create table experiments_alpha_4
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


create table models_alpha_4
(
	model_id serial not null,
	angle smallint not null,
	pressure_coefficients smallint[][] not null,
	
	constraint FK_d1b9c80f_2df9_4cee_8131_873ae701025f foreign key (model_id) references experiments_alpha_4(model_id)
);


insert into models_alpha_4
values
(1,1, array[[212,2],[1,2]]);

select * from models_alpha_4


insert into models_4 (model_name,breadth, depth, height, sample_frequency, sample_period, uh_AverageWindSpeed, angle, x_coordinates, y_coordinates, z_coordinates, pressure_coefficients)
values
(111,0.1,0.2,0.3,1000,11.57,3.4,10,array[0.3,0.2],array[0.1,0.4],array[0.2,0.5],array[array[10000,20000],array[10230,20100],array[10006,20800]])




drop table alpha_41
create table models_41
(
	model_id serial,
	model_name smallint,
	
	constraint FK_model_name foreign key (model_name) references alpha_4(model_name)
)

insert into models_41(model_name)
values
(116)

select * from models_4

select * from alpha_4
left join models_4 using(model_name)




create table models_41
(
	model_id serial,
	pressure_coefficients smallint[][]
)

insert into models_4 (pressure_coefficients)
values
({array[10000,20000],array[10230,20100],array[10006,20800]])






insert into experiment (model_name)
values
(111)
returning model_id



select * from par



insert into par (model_id, breadth)
values
(8800549db20948a3970ac5b41a42af10,12.2)


select * from experiment;

create table experiment
(
	model_id uuid default uuid_generate_v4() primary key not null,
	model_name smallint not null
);

create table par
(
	model_id uuid not null,
	breadth real not null,
	
	constraint FK_733f1de5_ed90_458e_947 foreign key (model_id) references experiment(model_id)
);


insert into par (breadth)
values
(8800549d-b209-48a3-970a-c5b41a42af10,12.2)








drop table e
create table e
(
	model_id serial primary key,
	model_name smallint not null
);

insert into e (model_name)
values
(123),
(124),
(125)


create table par1
(
	model_id serial not null,
	breadth real not null,
	
	constraint FK_ed90_458e_947 foreign key (model_id) references e(model_id)
);


insert into par1
values
(1,12),
(2,11),
(3,9)

select * from par1
