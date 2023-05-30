import pickle
import inflection
import pandas as pd
import numpy as np
import math
import requests
import datetime



class Rossmann(object):
    def __init__(self):
        self.home_path = ''
        self.competition_distance_scaler   = pickle.load(open(self.home_path + 'parameter/competition_distance_scaler.pkl', 'rb'))
        self.competition_time_month_scaler = pickle.load(open(self.home_path + 'parameter/competition_time_month_scaler.pkl', 'rb'))
        self.promo_time_week_scaler        = pickle.load(open(self.home_path + 'parameter/promo_time_week_scaler.pkl', 'rb'))
        self.year_scaler                   = pickle.load(open(self.home_path + 'parameter/year_scaler.pkl', 'rb'))
        self.store_type_scaler             = pickle.load(open(self.home_path + 'parameter/store_type_scaler.pkl', 'rb'))

    def data_cleaning(self, df1):

        ## 1 RENOMEANDO COLUNAS
        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo','StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
                    'CompetitionDistance', 'CompetitionOpenSinceMonth','CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
                    'Promo2SinceYear', 'PromoInterval']

        #função p/ transformar os nomes em separados por underline
        snakecase = lambda x: inflection.underscore(x).lower()

        #nova lista trocando o nome old pelo new
        cols_new = list(map(snakecase, cols_old))

        #rename old columns
        df1.columns = cols_new

        #transformando date
        df1['date'] = pd.to_datetime(df1['date'])

        #competiton_distance
        #20000 - corresponde a um valor muito maior queo maximo, apenas para servir de outlier
        df1['competition_distance'] = df1['competition_distance'].apply (lambda x: 200000.0 if math.isnan(x)
                                                                                    else x)
                
        #competition_open_since_month
        #extract month
        df1['competition_open_since_month'] = df1.apply(lambda x: x['date'].month if math.isnan(x['competition_open_since_month'])
                                                                                else x['competition_open_since_month'], axis=1)

        #competition_open_since_year
        #extract year
        df1['competition_open_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['competition_open_since_year'])
                                                                                else x['competition_open_since_year'], axis=1)

        #promo2_since_week
        #extract year
        df1['promo2_since_week'] = df1.apply(lambda x: x['date'].week if math.isnan(x['promo2_since_week'])
                                                                    else x['promo2_since_week'], axis=1)

        #promo2_since_year
        df1['promo2_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['promo2_since_year'])
                                                                    else x['promo2_since_year'], axis=1)
        #promo_interval
        month_map = {1: 'Jan', 2: 'Fev',3: 'Mar',4: 'Apr',5: 'May',6: 'Jun',7: 'Jul',8: 'Aug',9: 'Sep',10: 'Oct',11: 'Nov',12: 'Dec'}

        #substituindo nulos por '0'
        df1['promo_interval'].fillna(0,inplace=True)

        #criando novas colunas p/ numero de mes ser substituido por letra
        df1['month_map'] = df1['date'].dt.month.map(month_map)

        #nova coluna com condições
        #CONDIÇÕES:
        #0 se o valor já for zero
        #1 se existir (month_map) no (promo_interval)
        #0 se não existir (month_map) no (promo_interval)
        df1['is_promo'] = df1[['promo_interval', 'month_map']].apply (lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split(',') else 0, axis=1)

        ## 3 ALTERANDO TIPO DE DADOS
        #change types
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(int)
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype(int)
        df1['promo2_since_week'] = df1['promo2_since_week'].astype(int)
        df1['promo2_since_year'] = df1['promo2_since_year'].astype(int)

        #int32 para int64
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype('int64')
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype('int64')
        df1['promo2_since_week'] = df1['promo2_since_week'].astype('int64')
        df1['promo2_since_year'] = df1['promo2_since_year'].astype('int64')

        return df1
    
    def feature_engineering(self, data):


        # year
        data['year'] = data['date'].dt.year

        # month
        data['month'] = data['date'].dt.month

        # day 
        data['day'] = data['date'].dt.day

        # week of year
        data['week_of_year'] = data['date'].dt.strftime('%U')

        # year week
        data['year_week'] = data['date'].dt.strftime('%Y-%W')

        # competition since
        data['competition_since'] = data.apply(lambda x: datetime.datetime(year=x['competition_open_since_year'], month=x['competition_open_since_month'], day=1), axis=1)
        data['competition_time_month'] = ((data['date'] - data['competition_since']) / 30).apply(lambda x: x.days).astype(int)

        # promo since
        data['promo_since'] = data['promo2_since_year'].astype(str) + '-' + data['promo2_since_week'].astype(str)
        data['promo_since'] = data['promo_since'].apply(lambda x:datetime.datetime.strptime(x + '-1', '%Y-%W-%w') - datetime.timedelta(days = 7))
        data['promo_time_week'] = ((data['date'] - data['promo_since'])/7).apply(lambda x: x.days).astype(int)


        # assortment
        data['assortment'] = data['assortment'].apply(lambda x: 'basic' if x == 'a'
                                                        else 'extra' if x == 'b' 
                                                        else 'extended')

        # state holiday
        data['state_holiday'] = data['state_holiday'].apply(lambda x: 'public_holiday' if x == 'a'
                                                        else 'easter_holiday' if x == 'b'
                                                        else 'christmas' if x == 'c'
                                                        else 'regular_day')

        ## 2.3 FILTRAGEM DE VARIAVEIS

        ### Filtragem das Linhas
        #filtrando linhas em que a loja estava aberta e as vendas foram maior que 0
        data = data[data['open'] != 0]
        ### Seleção das Colunas
        #excluindo colunas em que serviram para criação variaveis
        #excluindo 'customers', pois não serve para o modelo
        cols_drop = ['open', 'promo_interval', 'month_map']
        data = data.drop(cols_drop, axis=1)

        return data
    

    def data_preparation(self, df4):

        # competition distance
        df4['competition_distance'] = self.competition_distance_scaler.fit_transform(df4[['competition_distance']].values)

        # competition time month
        df4['competition_time_month'] = self.competition_time_month_scaler.fit_transform(df4[['competition_time_month']].values)
        
        # promo time week
        df4['promo_time_week'] = self.promo_time_week_scaler.fit_transform(df4[['promo_time_week']].values)

        #year
        df4['year'] = self.year_scaler.fit_transform(df4[['year']].values)


        ### Encoding
        #state_holiday - ONE HOT ENCODING -> divide em novas colunas com nome das linhas unicas
        df4 = pd.get_dummies(df4, prefix=['state_holiday'], columns=['state_holiday'])

        """#transformando booleanos em int
        #boolean_dict = {True: 1,
        #                False: 0}

        #df4['state_holiday_christmas'] = df4['state_holiday_christmas'].map(boolean_dict)
        #df4['state_holiday_easter_holiday'] = df4['state_holiday_easter_holiday'].map(boolean_dict)
        #df4['state_holiday_public_holiday'] = df4['state_holiday_public_holiday'].map(boolean_dict)
        #df4['state_holiday_regular_day'] = df4['state_holiday_regular_day'].map(boolean_dict)"""

        #store_type - LABEL ENCODING -> tranforma coluna que antes era nome, em numero.
        #utilizado nessa situação, pois não existe uma hierarquia na diferença desses dados (a , b, c, d)
        df4['store_type'] = self.store_type_scaler.fit_transform(df4['store_type'])

        #assortment - ORDINAL ENCODING -> ordena por tamanho, necessario entendimento de negocio p/ passar um dicionario
        assortment_dict = {'basic': 1,
                           'extra': 2,
                           'extended': 3}

        df4['assortment'] = df4['assortment'].map(assortment_dict)
        
        ### Nuture Transformation
        #day o week
        df4['day_of_week_sin'] = df4['day_of_week'].apply(lambda x: np.sin(x * ( 2. * np.pi/7) ) )
        df4['day_of_week_cos'] = df4['day_of_week'].apply(lambda x: np.cos(x * ( 2. * np.pi/7) ) )

        #month
        df4['month_sin'] = df4['month'].apply(lambda x: np.sin(x * ( 2. * np.pi/12) ) )
        df4['month_cos'] = df4['month'].apply(lambda x: np.cos(x * ( 2. * np.pi/12) ) )

        #day
        df4['day_sin'] = df4['day'].apply(lambda x: np.sin(x * ( 2. * np.pi/30) ) )
        df4['day_cos'] = df4['day'].apply(lambda x: np.cos(x * ( 2. * np.pi/30) ) )

        #week of year
        df4['week_of_year_sin'] = df4['week_of_year'].apply(lambda x: np.sin(float(x) * ( 2. * np.pi/52) ) )
        df4['week_of_year_cos'] = df4['week_of_year'].apply(lambda x: np.cos(float(x) * ( 2. * np.pi/52) ) )

        cols_selected = ['store','promo','store_type','assortment','competition_distance','competition_open_since_month',
                                'competition_open_since_year','promo2','promo2_since_week','promo2_since_year','competition_time_month',
                                'promo_time_week','day_of_week_sin','day_of_week_cos','month_sin','month_cos',
                                'day_sin','day_cos','week_of_year_cos','week_of_year_sin']


        return df4[cols_selected]
    
    def get_predction(self, model, original_data, test_data):
        #prediction
        pred = model.predict(test_data)

        #join pred into the original data
        original_data['predction'] = np.expm1(pred)

        return original_data.to_json(orient='records', date_format='iso')
    