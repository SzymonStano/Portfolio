import pandas as pd
import numpy as np
import os

from sqlalchemy import Column, Integer, String, ForeignKey, Date
from sqlalchemy.orm import relationship
from models.base import Base

class Kupno_Sprzedaz(Base):
    __tablename__ = 'Kupno_Sprzedaz'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    typ = Column(String(50), index=True)
    id_klienta = Column(Integer, ForeignKey('Klienci.id'), nullable=False)
    id_pojazdu = Column(Integer, ForeignKey('Renowacje_Pojazdy.id'), nullable=False)
    kwota = Column(Integer, index=True)
    data = Column(Date, index=True)
    
    klient = relationship("Klienci", back_populates="kupno_sprzedaz")
    pojazd = relationship("Renowacje_Pojazdy", back_populates="kupno_sprzedaz")
    
class Kupno_SprzedazDB():
    def __init__(self, klienci_id, pojazdy, renowacje_df):
        self.klienci_id = klienci_id + 1
        self.renowacje_df = renowacje_df
        self.pojazdy = pojazdy
        
    def generate(self):
        self.df = pd.DataFrame(columns=['typ', 'id_klienta', 'id_pojazdu', 'kwota', 'data'])

        kupno_typ = ['kupno'] * len(self.renowacje_df)
        kupno_id_klienta = np.random.choice(self.klienci_id, len(self.renowacje_df))
        kupno_id_pojazdu = self.pojazdy.index + 1
        kupno_kwota = self.pojazdy['cena_sprzedaży'].values
        kupno_data = self.renowacje_df['data_rozpoczecia'].values

        kupno_df = pd.DataFrame({
            'typ': kupno_typ,
            'id_klienta': kupno_id_klienta,
            'id_pojazdu': kupno_id_pojazdu,
            'kwota': kupno_kwota,
            'data': kupno_data
        })

        renowacje_filtered = self.renowacje_df[self.renowacje_df['data_zakonczenia'].notna()]

        sprzedaz_typ = ['sprzedaż'] * len(renowacje_filtered)
        sprzedaz_id_klienta = np.random.choice(self.klienci_id, len(renowacje_filtered))
        sprzedaz_id_pojazdu = renowacje_filtered['id_pojazdu']
        sprzedaz_kwota = self.pojazdy.loc[renowacje_filtered['id_pojazdu']-1, 'cena_sprzedaży'].to_numpy() * np.random.uniform(1.2, 1.4, len(renowacje_filtered))
        sprzedaz_data = renowacje_filtered['data_zakonczenia'].to_numpy()

        sprzedaz_df = pd.DataFrame({
            'typ': sprzedaz_typ,
            'id_klienta': sprzedaz_id_klienta,
            'id_pojazdu': sprzedaz_id_pojazdu,
            'kwota': sprzedaz_kwota,
            'data': sprzedaz_data
        })

        self.df = pd.concat([kupno_df, sprzedaz_df], ignore_index=True)
        self.df = self.df.replace({np.nan: None})

        return self.df