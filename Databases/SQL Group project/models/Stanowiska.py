import pandas as pd

from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from models.base import Base

class Stanowiska(Base):
    __tablename__ = 'Stanowiska'
    id = Column(Integer, primary_key=True, autoincrement=True)
    nazwa = Column(String(50), nullable=False, unique=True)
    pensja_min = Column(Integer, nullable=False)
    pensja_max = Column(Integer, nullable=False)
    
    pracownik = relationship("Pracownicy", back_populates="stanowisko")

class StanowiskaDB:
    def __init__(self):
        self.df = pd.read_csv('data/stanowiska.csv')

    def generate(self):
        return self.df
