from sqlalchemy import Column, Integer, String, Float, ForeignKey
from models.base import Base
import pandas as pd
from sqlalchemy.orm import relationship

class Uslugi(Base):
    __tablename__ = 'Uslugi'

    id = Column(Integer, primary_key=True, index=True)
    nazwa_uslugi = Column(String(100))
    typ_uslugi = Column(String(50))
    naprawa_renowacja = Column(String(50))
    stanowisko = Column(Integer)
    cena_dolna = Column(Float)
    cena_gorna = Column(Float)

    uslugi_czesci = relationship("Uslugi_Czesci", back_populates="usluga")
    naprawa = relationship("Uslugi_Napraw", back_populates="usluga")
    uslugi_renowacji = relationship("Uslugi_Renowacji", back_populates="usluga")

class UslugiDB:
    def __init__(self, xlsx_file='data/uslugi.xlsx'):
        self.df = pd.read_excel(xlsx_file)

    def generate(self):
        return self.df
