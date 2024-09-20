import pandas as pd

from sqlalchemy import Column, Integer, String
from models.base import Base
from sqlalchemy.orm import relationship

class Kraje(Base):
    __tablename__ = 'Kraje'

    id = Column(Integer, primary_key=True, index=True)
    nazwa = Column(String(50), index=True)
    
    miejscowosci = relationship("Miejscowosci", back_populates="kraj")

class KrajeDB():
    def __init__(self):
        self.countries = pd.read_csv('data/kraje.csv')
    
    def generate(self):
        return self.countries
