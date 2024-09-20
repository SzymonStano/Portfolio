import pandas as pd
import numpy as np
import models.Klienci as Klienci

from sqlalchemy import Column, Integer, String, Date, ForeignKey
from sqlalchemy.orm import relationship
from models.base import Base

class Pracownicy(Base):
    __tablename__ = "Pracownicy"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    imie = Column(String(50), index=True)
    nazwisko = Column(String(50), index=True)
    data_urodzenia = Column(Date, index=True)
    nr_telefonu = Column(String(15), index=True)
    email = Column(String(50), index=True)
    id_adresu = Column(Integer, ForeignKey('Adresy.id'), nullable=False)
    id_stanowiska = Column(Integer, ForeignKey('Stanowiska.id'), nullable=False)
    pensja = Column(Integer, index=True)

    adres = relationship("Adresy", back_populates="pracownik")
    stanowisko = relationship("Stanowiska", back_populates="pracownik")
    usluga_napraw = relationship("Uslugi_Napraw", back_populates="pracownik")
    usluga_renowacji = relationship("Uslugi_Renowacji", back_populates="pracownik")

    
class PracownicyDB():
    def __init__(self, adresses_ID, stanowiska_df):
        
        self.IDs = adresses_ID
        self.data, _ = Klienci.KlienciDB(size=adresses_ID.shape[0], employees_size=0).generate()
        self.stanowiska = stanowiska_df
    
    def generate(self):
        stanow = [1, 1, 1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10, 11, 11, 12, 12, 12]
        id_stanowiska = np.random.choice(stanow, size=len(stanow), replace=False)
        
        self.data['id_adresu'] = self.IDs
        self.data['id_stanowiska'] = id_stanowiska
        
        df_stanowiska = pd.DataFrame({'id_stanowiska': id_stanowiska})
        merged_df = df_stanowiska.merge(self.stanowiska, left_on='id_stanowiska', right_on='Id', how='left')
        
        pensje_min = merged_df['Pensja_min'].to_numpy()
        pensje_max = merged_df['Pensja_max'].to_numpy()
        pensje = np.random.randint(pensje_min, pensje_max, size=len(stanow))
        pensje = pensje - (pensje % 500)
            
        self.data['pensja'] = pensje
        
        return self.data