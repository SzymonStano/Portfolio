import pandas as pd
import numpy as np
import datetime

from sqlalchemy import Column, Integer, String, Date, ForeignKey
from sqlalchemy.orm import relationship
from models.base import Base

class Uslugi_Napraw(Base):
    __tablename__ = 'Uslugi_Napraw'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    id_uslugi = Column(Integer, ForeignKey('Uslugi.id'), nullable=False)
    koszt_uslugi = Column(Integer, nullable=False)
    id_naprawy = Column(Integer, ForeignKey('Naprawy.id'), nullable=False)
    id_pracownika = Column(Integer, ForeignKey('Pracownicy.id'), nullable=False)
    
    usluga = relationship("Uslugi", back_populates="naprawa")
    naprawa = relationship("Naprawy", back_populates="usluga")
    pracownik = relationship("Pracownicy", back_populates="usluga_napraw")
    
class Uslugi_NaprawDB():
    def __init__(self, uslugi_lista, naprawy_id, pracownicy_df):
        self.uslugi_lista = uslugi_lista
        self.naprawy_id = naprawy_id + 1
        self.pracownicy_df = pracownicy_df
    
    def generate(self):
        uslugi_df = pd.read_excel('data/uslugi.xlsx')
        
        naprawy_id_flat = np.repeat(self.naprawy_id, [len(uslugi) for uslugi in self.uslugi_lista])
        uslugi_lista_flat = np.concatenate(self.uslugi_lista)
        
        price_ranges = uslugi_df.loc[uslugi_lista_flat - 1, ['Cena dolna', 'Cena gorna']].to_numpy()
        koszt_naprawy = np.array([np.random.randint(low, high) for low, high in price_ranges])

        stanowiska = uslugi_df.loc[uslugi_lista_flat - 1, 'Stanowisko'].to_numpy()
        pracownicy_indices = [self.pracownicy_df[self.pracownicy_df['id_stanowiska'] == stanowisko].index.to_numpy() for stanowisko in stanowiska]
        
        id_pracownika = np.array([np.random.choice(indices) for indices in pracownicy_indices])

        self.df = pd.DataFrame({
            'id_uslugi': uslugi_lista_flat,
            'koszt_uslugi': koszt_naprawy,
            'id_naprawy': naprawy_id_flat,
            'id_pracownika': id_pracownika + 1
        })
        return self.df