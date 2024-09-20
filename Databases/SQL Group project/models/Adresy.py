import pandas as pd
import numpy as np

from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from models.base import Base

class Adresy(Base):
    __tablename__ = 'Adresy'

    id = Column(Integer, primary_key=True, autoincrement=True)
    ulica = Column(String(50), index=True)
    nr_budynku = Column(String(10), index=True)
    nr_mieszkania = Column(String(10), index=True)
    kod_pocztowy = Column(String(10), index=True)
    id_miejscowosci = Column(Integer, ForeignKey('Miejscowosci.id'), nullable=False)

    miejscowosc = relationship("Miejscowosci", back_populates="adresy")
    klient = relationship("Klienci", back_populates="adres")
    pracownik = relationship('Pracownicy', back_populates='adres')
    


class AdresyDB():
    def __init__(self, size, cities_size):
        self.size = size
        self.cities_size = cities_size
        
        self.zips = pd.read_csv('data/uszips.csv')
        self.zips = self.zips[self.zips['state_name'].isin(['California'])]
        self.streets = pd.read_csv('data/ulice.csv').dropna()
        
        self.data = pd.DataFrame()
    
    def generate(self):
        streets = self.streets['Street Name'].values
        self.data['ulica'] = np.random.choice(streets, size=self.size)
        
        random_probs = np.random.rand(self.size)
        num_with_letters = np.char.add(
            np.random.randint(1, 100, size=self.size).astype(str),
            np.random.choice(['A', 'B', 'C', 'D', 'E', 'F'], size=self.size)
        )
        num_without_letters = np.random.randint(1, 500, size=self.size).astype(str)
        
        house_numbers = np.where(random_probs < 0.6, num_with_letters, num_without_letters)
        self.data['nr_budynku'] = house_numbers      
        
        flat_numbers = []
        random_probs = np.random.rand(self.size)
        flat_numbers = np.char.add(
            np.random.randint(1, 100, size=self.size).astype(str),
            np.random.choice(['A', 'B', 'C', 'D', 'E', 'F'], size=self.size)
        )
        flat_no_numbers = np.full(self.size, None)
        flats = np.where(random_probs < 0.6, flat_numbers, flat_no_numbers)
        self.data['nr_mieszkania'] = flats
        
        zips = self.zips['zip'].values
        self.data['kod_pocztowy'] = np.random.choice(zips, size=self.size)
        self.data['id_miejscowosci'] = np.random.randint(low=2, high=self.cities_size, size=self.size)
        self.data.index = np.arange(1, self.size+1)
        
        return self.data

        
    