from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.orm import relationship
from models.base import Base
import pandas as pd

class Czesci(Base):
    __tablename__ = 'Czesci'

    id = Column(Integer, primary_key=True, index=True)
    nazwa_czesci = Column(String(30))
    cena = Column(Float)
    
    czesci_uslugi = relationship("Uslugi_Czesci", back_populates="czesc")


class CzesciDB:
    # części zapisane akurat w pliku xlsx, ponieważ nie chciałem zmieniać wszystkich znaków polskich
    def __init__(self):
        self.df = pd.read_excel('data/czesci.xlsx')

    def generate(self):
        return self.df
