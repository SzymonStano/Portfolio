import pandas as pd
import numpy as np

from sqlalchemy import Column, Integer, String, Float, Boolean
from sqlalchemy.orm import relationship
from models.base import Base

class Naprawy_Pojazdy(Base):
    __tablename__ = 'Naprawy_Pojazdy'
    
    id = Column(Integer, primary_key=True, index=True)
    typ_pojazdu = Column(String(30))
    marka = Column(String(50))
    model = Column(String(100))
    rok_produkcji = Column(Integer)
    stan = Column(Integer)
    naped = Column(String(50))
    rodzaj_paliwa = Column(String(50))
    skrzynia_biegow = Column(String(100))
    przebieg = Column(Integer)
    
    naprawa = relationship("Naprawy", back_populates="pojazd")
    
class Naprawy_PojazdyDB:
    def __init__(self, cars_size, bikes_size):
        self.data = pd.read_csv('data/cars.csv').dropna(subset=['Identification.Make', 
                                                                'Identification.Model Year', 
                                                                'Engine Information.Driveline', 
                                                                'Engine Information.Transmission'])
        self.data = self.data.where(pd.notna(self.data), None)
        self.data = self.data.sample(n=cars_size)
        self.data = self.data.reset_index(drop=True)
        self.cars_size = cars_size
        
        self.data_bikes = pd.read_csv('data/bikes.csv', low_memory=False).dropna(subset=['Brand', 'Model', 'Year', 'Gearbox'])
        self.data_bikes = self.data_bikes.where(pd.notna(self.data_bikes), None)
        self.data_bikes = self.data_bikes.sample(n=bikes_size)
        self.data_bikes = self.data_bikes.reset_index(drop=True)
        self.bikes_size = bikes_size
        
    def adding_bikes(self):
        self.df_bikes = pd.DataFrame()
        self.df_bikes['typ_pojazdu'] = ['Motocykl'] * self.bikes_size
        self.df_bikes['marka'] = self.data_bikes['Brand']
        self.df_bikes['model'] = self.data_bikes['Model']
        self.df_bikes['rok_produkcji'] = self.data_bikes['Year']
        self.df_bikes['stan'] = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.10, 0.20, 0.30, 0.35], size=self.bikes_size)
        self.df_bikes['naped'] = None
        self.df_bikes['rodzaj_paliwa'] = np.random.choice(['Gasoline', 'Diesel', 'Electric', 'Hybrid'], p=[0.65, 0.2, 0.1, 0.05], size=self.bikes_size)
        self.df_bikes['skrzynia_biegow'] = self.data_bikes['Gearbox']
        self.df_bikes['przebieg'] = np.random.uniform(1000, 50_000, size=self.bikes_size).astype(int)

    def generate(self):
        self.df = pd.DataFrame()
        self.df['typ_pojazdu'] = ['Samochód'] * self.cars_size
        self.df['marka'] = self.data['Identification.Make']
        self.df['model'] = self.data['Identification.Model Year'].str[5:]
        self.df['rok_produkcji'] = np.random.choice(list(range(1995, 2025)), size=self.cars_size)
        self.df['stan'] = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.10, 0.20, 0.30, 0.35], size=self.cars_size)
        self.df['naped'] = self.data['Engine Information.Driveline']
        self.df['rodzaj_paliwa'] = np.random.choice(['Gasoline', 'Diesel', 'Electric', 'Hybrid'], p=[0.65, 0.2, 0.1, 0.05], size=self.cars_size)
        self.df['skrzynia_biegow'] = self.data['Engine Information.Transmission']
        self.df['przebieg'] = np.random.uniform(10000, 350000, size=self.cars_size).astype(int)

        self.adding_bikes()
        self.df = pd.concat([self.df, self.df_bikes])
        self.df = self.df.reset_index(drop=True)
        
        self.df = self.df.replace(np.nan, None)
        
        return self.df.sample(frac=1)
    