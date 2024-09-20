import pandas as pd
import numpy as np
import datetime

from sqlalchemy import Column, Integer, String, ForeignKey, Date
from sqlalchemy.orm import relationship
from models.base import Base

class Naprawy(Base):
    __tablename__ = 'Naprawy'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    id_klienta = Column(Integer, ForeignKey('Klienci.id'), nullable=False)
    id_pojazdu = Column(Integer, ForeignKey('Naprawy_Pojazdy.id'), nullable=False)
    data_rozpoczecia = Column(Date, index=True)
    data_zakonczenia = Column(Date, index=True)
    
    naprawa = relationship("Klienci", back_populates="klient")
    usluga = relationship("Uslugi_Napraw", back_populates="naprawa")
    pojazd = relationship("Naprawy_Pojazdy", back_populates="naprawa")
    
class NaprawyDB():
    def __init__(self, klienci_id, naprawy_pojazdy_index):
        self.klienci_id = klienci_id + 1
        self.naprawy_pojazdy_index = naprawy_pojazdy_index + 1
        self.size = naprawy_pojazdy_index.shape[0]
        
    def generate_uslugi(self):
        uslugi = pd.read_excel('data/uslugi.xlsx')
        uslugi = uslugi[uslugi['Naprawa/Renowacja'].isin(['Naprawa', 'Naprawa/Renowacja'])]
        
        ilosc_uslug = np.random.randint(low=1, high=5, size=self.size)
        uslugi_lista = [np.random.choice(uslugi['ID'], size=ilosc, replace=False) for ilosc in ilosc_uslug]
        
        return uslugi_lista

    def generate(self):
        uslugi_lista = self.generate_uslugi()
        today = pd.Timestamp(datetime.datetime.now().date())
        self.df = pd.DataFrame(columns=['id_klienta','id_pojazdu','data_rozpoczecia','data_zakonczenia'])
        
        id_klienta_array = np.random.choice(self.klienci_id, size=self.size, replace=True)
        id_pojazdu_array = self.naprawy_pojazdy_index
        
        start_dates = today - pd.to_timedelta(np.random.randint(0, 2*365, size=self.size), unit='D')
        
        k_array = np.array([len(uslugi) for uslugi in uslugi_lista])
        multipliers = 4 * (np.log(k_array**2)) + 4
        means = multipliers
        sigmas = multipliers / 6
        
        end_dates_offsets = np.ceil(np.abs(np.random.normal(means, sigmas)))
        end_dates = start_dates + pd.to_timedelta(end_dates_offsets, unit='d')
        
        self.df = pd.DataFrame({
            'id_klienta': id_klienta_array,
            'id_pojazdu': id_pojazdu_array,
            'data_rozpoczecia': start_dates,
            'data_zakonczenia': end_dates
        })
        
        self.df['data_zakonczenia'] = self.df['data_zakonczenia'].apply(lambda x: x if x < today else None)
    

        self.df = self.df.replace({pd.NaT: None})
        return self.df, uslugi_lista
    
# xd = NaprawyDB(np.arange(1,100), np.arange(1,100))
# print(xd.generate())