import pandas as pd

from sqlalchemy import Column, Integer, ForeignKey
from sqlalchemy.orm import relationship
from models.base import Base


class Uslugi_Czesci(Base):
    __tablename__ = 'Uslugi_Czesci'

    id = Column(Integer, primary_key=True, index=True)
    id_uslugi = Column(Integer, ForeignKey('Uslugi.id'))
    id_czesci = Column(Integer, ForeignKey('Czesci.id'))

    usluga = relationship("Uslugi", back_populates="uslugi_czesci")
    czesc = relationship("Czesci", back_populates="czesci_uslugi")


class Uslugi_CzesciDB:
    def __init__(self):
        self.df = pd.read_excel("data/uslugi_czesci.xlsx")
        self.df.replace(0, None, inplace=True)

    def generate(self):
        return self.df
