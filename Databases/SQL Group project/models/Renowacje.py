import pandas as pd
import numpy as np
import datetime

from sqlalchemy import Column, Integer, String, ForeignKey, Date
from sqlalchemy.orm import relationship
from models.base import Base


class Renowacje(Base):
    __tablename__ = 'Renowacje'

    id = Column(Integer, primary_key=True, autoincrement=True)
    id_pojazdu = Column(Integer, ForeignKey('Renowacje_Pojazdy.id'), nullable=False)
    data_rozpoczecia = Column(Date, index=True)
    data_zakonczenia = Column(Date, index=True)

    pojazd = relationship("Renowacje_Pojazdy", back_populates="renowacja")
    usluga = relationship("Uslugi_Renowacji", back_populates="renowacja")


class RenowacjeDB():
    def __init__(self, pojazdy):
        self.pojazdy = pojazdy

    def generate_uslugi(self):
        uslugi = pd.read_excel('data/uslugi.xlsx')
        uslugi = uslugi[uslugi['Naprawa/Renowacja'].isin(['Renowacja', 'Naprawa/Renowacja'])]
        
        ilosc_uslug = np.random.randint(low=1, high=5, size=len(self.pojazdy))
        uslugi_lista = [np.random.choice(uslugi['ID'], size=ilosc, replace=False) for ilosc in ilosc_uslug]
        
        return uslugi_lista

    def generate(self):
        uslugi_lista = self.generate_uslugi()
        num_pojazdy = len(self.pojazdy)
        
        id_pojazdu = self.pojazdy.index
        
        today = pd.Timestamp(datetime.datetime.now().date())
        start_dates = today - pd.to_timedelta(np.random.randint(60, 2*365, size=num_pojazdy), unit='d')
        
        k_array = np.array([len(uslugi) for uslugi in uslugi_lista])
        multiplier = 4 * (np.log(k_array**2)) + 4
        mean = multiplier
        sigma = multiplier / 6
        
        end_dates = np.full(num_pojazdy, None)
        
        for i in id_pojazdu:
            if self.pojazdy.iloc[i]:
                while True:
                    end = start_dates[i] + pd.to_timedelta(np.ceil(np.abs(np.random.normal(mean[i], sigma[i]))), unit='d')
                    if end <= today:
                        end_dates[i] = end
                        break
        
        # Create DataFrame
        self.df = pd.DataFrame({
            'id_pojazdu': id_pojazdu + 1,
            'data_rozpoczecia': start_dates,
            'data_zakonczenia': end_dates
        })
        
        self.df = self.df.replace({pd.NaT: None})
        
        return self.df, uslugi_lista

