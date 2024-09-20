import pandas as pd
import numpy as np
import datetime

from sqlalchemy import Column, Integer, String, Date, ForeignKey
from sqlalchemy.orm import relationship
from models.base import Base

class Wyposazenie(Base):
    __tablename__ = 'Wyposazenie'
    
    id = Column(Integer, primary_key=True)
    nazwa = Column(String(50))
    typ_przeznaczenia = Column(String(50))
    cena = Column(Integer)
    
class WyposazenieDB:
    def __init__(self):
        self.data = pd.read_excel('data/wyposazenie.xlsx')
        
    def generate(self):
        low = self.data['cena_dolna'].to_numpy()
        high = self.data['cena_górna'].to_numpy()
        self.data['cena'] = np.mean([low, high], axis=0)
        return self.data[['id_wyposazenia', 'nazwa', 'typ_przeznaczenia', 'cena']]
                