import pandas as pd
import numpy as np
import datetime

from sqlalchemy import Column, Integer, String, Date, ForeignKey
from sqlalchemy.orm import relationship
from models.base import Base

class Klienci(Base):
    __tablename__ = 'Klienci'

    id = Column(Integer, primary_key=True, autoincrement=True)
    imie = Column(String(50), index=True)
    nazwisko = Column(String(50), index=True)
    data_urodzenia = Column(Date, index=True)
    nr_telefonu = Column(String(15), index=True)
    email = Column(String(50), index=True)
    id_adresu = Column(Integer, ForeignKey('Adresy.id'), nullable=False)

    adres = relationship("Adresy", back_populates="klient")
    klient = relationship("Naprawy", back_populates="naprawa")
    kupno_sprzedaz = relationship("Kupno_Sprzedaz", back_populates="klient")

class KlienciDB():
    def __init__(self, size, employees_size):
        
        self.df = pd.DataFrame()
        self.surnames = pd.read_csv('data/nazwiska.csv').dropna()
        self.names = pd.read_csv('data/imiona.csv').dropna()
        self.cities = pd.read_csv('data/uszips.csv')
        self.cities_CA = self.cities[self.cities['state_name'].isin(['California'])]
        self.cities_rest = self.cities[self.cities['state_name'].isin(['Nevada', 'Oregon', 'Arizona'])]
        self.streets = pd.read_csv('data/ulice.csv').dropna()
        self.size = size
        self.employees = employees_size

    def generate_surname(self):
        probabilities = self.surnames['count'].values / self.surnames['count'].sum()
        surnames = self.surnames['name'].values
        return np.random.choice(surnames, p=probabilities, size=self.size)
    
    def generate_name(self):
        names = self.names['Name'].apply(lambda x: x.upper()).values
        probabilities = self.names['Frequency'].values / self.names['Frequency'].sum()
        return np.random.choice(names, p=probabilities, size=self.size)
    
    def generate_age(self):
        ages = np.random.normal(37, 12, self.size)   
        ages = np.round(ages)
        ages = np.clip(ages, 18, 101) 
        
        birth_dates = []
        for age in ages:
            month = np.random.randint(low=1, high=13)
            year = int(2024 - age)
            if month == 2:
                day = np.random.randint(low=1, high=29)
            elif month in [4, 6, 9, 11]:
                day = np.random.randint(low=1, high=31)
            else:
                day = np.random.randint(low=1, high=32)
            birth_dates.append(datetime.date(year, month, day))
        return birth_dates
    
    def generate_phone(self):
        starts = np.array(['209', '213', '310', '323', '408', '415', '424', '442', '510', '530', '559', '562', '619', '626', 
                   '650', '657', '661', '669', '707', '714', '747', '760', '805', '818', '831', '858', '909', '916', 
                   '925', '949', '951'])

        start_indices = np.random.choice(len(starts), self.size)
        selected_starts = starts[start_indices]

        random_digits = np.random.randint(0, 10, (self.size, 7))
        random_digit_strings = np.apply_along_axis(lambda x: ''.join(x.astype(str)), 1, random_digits)

        numbers = np.char.add(selected_starts, random_digit_strings)
        return numbers
    
    def generate_email(self):
        emails = (self.df['imie'].str.lower() + '.' + self.df['nazwisko'].str.lower() + '@pimpmywheels.com').tolist()
        return emails


    def generate(self):
        self.df['imie'] = self.generate_name()
        self.df['nazwisko'] = self.generate_surname()
        self.df['data_urodzenia'] = self.generate_age()
        self.df['nr_telefonu'] = self.generate_phone()
        self.df['przedrostek_maila'] = self.df['imie'].str.lower() + '.' + self.df['nazwisko'].str.lower()
        
        self.df['duplikaty'] = self.df.groupby('przedrostek_maila').cumcount() + 1
        self.df['email'] = self.df.apply(
            lambda row: f"{row['przedrostek_maila']}{row['duplikaty']}@pimpmywheels.com" if row['duplikaty'] > 1 else f"{row['przedrostek_maila']}@pimpmywheels.com", 
            axis=1
        )
        self.df.drop(columns=['przedrostek_maila', 'duplikaty'], inplace=True)
        
        id_wszystkie = np.arange(1, self.size+self.employees+1)
        id_adresu = np.random.choice(id_wszystkie, replace=False, size=self.size)
        id_pracownikow = np.setdiff1d(id_wszystkie, id_adresu)
        self.df['id_adresu'] = id_adresu

        return self.df, id_pracownikow