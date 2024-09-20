import os
import pandas as pd
import numpy as np

from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from models.base import Base

class Miejscowosci(Base):
    __tablename__ = 'Miejscowosci'

    id = Column(Integer, primary_key=True, index=True)
    nazwa = Column(String(50), index=True)
    id_kraju = Column(Integer, ForeignKey('Kraje.id'), nullable=False)

    kraj = relationship("Kraje", back_populates="miejscowosci")
    adresy = relationship("Adresy", back_populates="miejscowosc")


class MiejscowosciDB():
    def __init__(self, size, percentage_CA):
        
        self.size = size
        self.percentage_CA = percentage_CA
        
        self.cities = pd.read_csv('data/uszips.csv')
        self.cities_CA = self.cities[self.cities['state_name'].isin(['California'])]
        self.cities_rest = self.cities[self.cities['state_name'].isin(['Nevada', 'Oregon', 'Arizona'])]
    
    def generate(self):
        cities_CA = self.cities_CA[['city', 'state_name']].values
        probabilities_CA = self.cities_CA['population'].values / self.cities_CA['population'].sum()
        
        cities_rest = self.cities_rest[['city', 'state_name']].values
        probabilities_rest = self.cities_rest['population'].values / self.cities_rest['population'].sum()

        random_CA_indices = np.random.choice(np.arange(len(cities_CA)), p=probabilities_CA, size=int(self.size * self.percentage_CA))
        random_rest_indices = np.random.choice(np.arange(len(cities_rest)), p=probabilities_rest, size=self.size - len(random_CA_indices))
        
        cities = np.concatenate((cities_CA[random_CA_indices], cities_rest[random_rest_indices]))
        cities = np.unique(cities.astype("<U22"), axis=0)
        
        self.data = pd.DataFrame(cities, columns=['nazwa', 'nazwa_stanu'], index=np.arange(1, cities.shape[0]+1))
        self.data['id_kraju'] = 159
        return self.data
