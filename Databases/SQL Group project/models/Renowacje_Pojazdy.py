from sqlalchemy import Column, Integer, String, Float, Boolean
from sqlalchemy.orm import relationship
from models.base import Base

import pandas as pd
import numpy as np

class Renowacje_Pojazdy(Base):
    __tablename__ = 'Renowacje_Pojazdy'

    id = Column(Integer, primary_key=True, index=True)
    typ_pojazdu = Column(String(30))
    marka = Column(String(50))
    model = Column(String(100))
    rok_produkcji = Column(Integer)
    konie_mechaniczne = Column(Integer)
    czy_sprzedane = Column(Boolean)
    stan = Column(Integer)
    naped = Column(String(50))
    rodzaj_paliwa = Column(String(50))
    skrzynia_biegow = Column(String(100))
    przebieg = Column(Integer)
    
    renowacja = relationship("Renowacje", back_populates="pojazd")
    kupno_sprzedaz = relationship("Kupno_Sprzedaz", back_populates="pojazd")

class Renowacje_PojazdyDB:
    def __init__(self, cars_size: int, bikes_size: int, percentage_cars_sold: float, percentage_bikes_sold: float):
        self.cars_data = pd.read_csv('data/cars.csv').dropna(subset=['Identification.Make',
                                                                'Identification.Model Year',
                                                                'Engine Information.Engine Statistics.Horsepower',
                                                                'Engine Information.Driveline',
                                                                'Engine Information.Transmission']).sample(n = cars_size)
        self.cars_data = self.cars_data.reset_index(drop=True)
        self.cars_size = cars_size
        
        self.percentage_cars_sold = percentage_cars_sold
        self.percentage_bikes_sold = percentage_bikes_sold
        
        self.bikes_data = pd.read_csv('data/bikes.csv', low_memory=False).dropna(subset=['Brand', 
                                                                                         'Model', 
                                                                                         'Year', 
                                                                                         'Power (hp)', 
                                                                                         'Gearbox']).sample(n = bikes_size)
        self.bikes_data = self.bikes_data.reset_index(drop=True)
        self.bikes_size = bikes_size
      
    
    def adding_bikes(self):
        self.df_bikes = pd.DataFrame()
        self.df_bikes['typ_pojazdu'] = ['Motocykl'] * self.bikes_size
        self.df_bikes['marka'] = self.bikes_data['Brand']
        self.df_bikes['model'] = self.bikes_data['Model']
        self.df_bikes['rok_produkcji'] = self.bikes_data['Year']
        self.df_bikes['konie_mechaniczne'] = pd.to_numeric(self.bikes_data['Power (hp)'], errors='coerce', downcast='integer')
        self.df_bikes['czy_sprzedane'] = np.random.choice([False, True], p=[1-self.percentage_bikes_sold, self.percentage_bikes_sold], size=self.bikes_size)
        self.df_bikes['stan'] = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.10, 0.20, 0.30, 0.35], size=self.bikes_size)
        self.df_bikes['naped'] = None
        self.df_bikes['rodzaj_paliwa'] = np.random.choice(['Gasoline', 'Diesel', 'Electric', 'Hybrid'], p=[0.65, 0.2, 0.1, 0.05], size=self.bikes_size)
        self.df_bikes['skrzynia_biegow'] = self.bikes_data['Gearbox']
        self.df_bikes['przebieg'] = np.random.uniform(150, 80_000, size=self.bikes_size).astype(int)
        self.df_bikes['cena_detaliczna'] = pd.to_numeric(self.bikes_data['Power (hp)'] * 1000 * (1 - (2024 - self.bikes_data['Year']) / 1000), errors='coerce').clip(lower=0).fillna(0).astype(int)
        self.df_bikes['cena_sprzedaży'] = self.calculate_car_value(self.df_bikes[['rok_produkcji','cena_detaliczna','marka','przebieg','stan']])

    def generate(self):
        self.df = pd.DataFrame()
        self.df['typ_pojazdu'] = ['Samochód'] * self.cars_size
        self.df['marka'] = self.cars_data['Identification.Make']
        self.df['model'] = self.cars_data['Identification.Model Year'].str[5:]
        self.df['rok_produkcji'] = np.random.choice(list(range(1995, 2025)), size=self.cars_size)
        self.df['konie_mechaniczne'] = self.cars_data['Engine Information.Engine Statistics.Horsepower']
        self.df['czy_sprzedane'] = np.random.choice([False, True], p=[1-self.percentage_cars_sold, self.percentage_cars_sold], size=self.cars_size)
        self.df['stan'] = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.10, 0.20, 0.30, 0.35], size=self.cars_size)
        self.df['naped'] = self.cars_data['Engine Information.Driveline']
        self.df['rodzaj_paliwa'] = np.random.choice(['Gasoline', 'Diesel', 'Electric', 'Hybrid'], p=[0.65, 0.2, 0.1, 0.05], size=self.cars_size)
        self.df['skrzynia_biegow'] = self.cars_data['Engine Information.Transmission']
        self.df['przebieg'] = np.random.uniform(50000, 350000, size=self.cars_size).astype(int)
        self.df['cena_detaliczna'] = pd.to_numeric(self.df['konie_mechaniczne'] * 1000 * (1 - (2024 - self.df['rok_produkcji']) / 1000), errors='coerce', downcast='integer').clip(lower=0).fillna(0).astype(int)
        self.df['cena_sprzedaży'] = self.calculate_car_value(self.df[['rok_produkcji','cena_detaliczna','marka','przebieg','stan']])

        self.adding_bikes()
        self.df = pd.concat([self.df, self.df_bikes])
        self.df = self.df.reset_index(drop=True)
        
        self.df = self.df.replace(np.nan, None)
        
        return self.df.sample(frac=1)
    
    
    @staticmethod  
    def calculate_car_value(cars_data):
        production_year = cars_data['rok_produkcji']
        retail_price = cars_data['cena_detaliczna']
        brand = cars_data['marka']
        mileage = cars_data['przebieg']
        condition = cars_data['stan']
        
        brand_depreciation_rate = {
            "Toyota": 0.15,
            "Honda": 0.15,
            "Ford": 0.20,
            "BMW": 0.25,
            "Mercedes": 0.25,
            "Other": 0.20
        }

        depreciation_rate = brand.apply(lambda x: brand_depreciation_rate.get(x, 0.20))
        car_age = 2024 - production_year
        depreciated_value = retail_price * ((1 - depreciation_rate) ** (car_age/10))
        mileage_depreciation = np.where(mileage > 100_000, (mileage - 100_000) / 10_000 * 0.001, 0)
        mileage_adjusted_value = depreciated_value * (1 - mileage_depreciation)
        condition_adjustment = {1: 0.5, 2: 0.7, 3: 0.85, 4: 0.95, 5: 1.0}
        condition_factor = condition.map(condition_adjustment)
        final_value = mileage_adjusted_value * condition_factor

        return pd.to_numeric(final_value.clip(lower=0), errors='coerce').clip(lower=0).fillna(0).astype(int)
