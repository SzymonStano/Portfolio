from models.base import Base, engine, SessionLocal
from models.Kraje import Kraje, KrajeDB
from models.Miejscowosci import Miejscowosci, MiejscowosciDB
from models.Klienci import Klienci, KlienciDB
from models.Adresy import Adresy, AdresyDB
from models.Stanowiska import Stanowiska, StanowiskaDB  # Import Stanowiska
from models.Pracownicy import Pracownicy, PracownicyDB
from models.Czesci import Czesci, CzesciDB  # Import Czesci
from models.Uslugi import Uslugi, UslugiDB  # Import Czesci
from models.Uslugi_Czesci import Uslugi_Czesci, Uslugi_CzesciDB
from models.Renowacje_Pojazdy import Renowacje_Pojazdy, Renowacje_PojazdyDB
from models.Wyposazenie import Wyposazenie, WyposazenieDB
from models.Renowacje import Renowacje, RenowacjeDB
from models.Kupno_Sprzedaz import Kupno_Sprzedaz, Kupno_SprzedazDB
from models.Naprawy import Naprawy, NaprawyDB
from models.Uslugi_Napraw import Uslugi_Napraw, Uslugi_NaprawDB
from models.Uslugi_Renowacji import Uslugi_Renowacji, Uslugi_RenowacjiDB
from models.Naprawy_Pojazdy import Naprawy_Pojazdy, Naprawy_PojazdyDB

import pandas as pd
import numpy as np


def load_data(size=1758, employees_size=18, cars_size_reno=157, bikes_size_reno=101, percentage_cars_sold=0.95, percentage_bikes_sold=0.95):
    kraje_df = KrajeDB().generate()
    miejscowosci_df = MiejscowosciDB(size+employees_size, 0.9).generate()
    adresy_df = AdresyDB(size+employees_size, miejscowosci_df.shape[0]).generate()
    klienci_df, id_pracownikow = KlienciDB(size, employees_size).generate()
    stanowiska_df = StanowiskaDB().generate()  # Load Stanowiska data
    pracownicy_df = PracownicyDB(id_pracownikow, stanowiska_df).generate()
    czesci_df = CzesciDB().generate()
    uslugi_df = UslugiDB().generate()
    uslugi_czesci_df = Uslugi_CzesciDB().generate()
    wyposazenie_df = WyposazenieDB().generate()
    
    renowacje_pojazdy_df = Renowacje_PojazdyDB(cars_size=cars_size_reno, 
                                               bikes_size=bikes_size_reno, 
                                               percentage_cars_sold=percentage_bikes_sold, 
                                               percentage_bikes_sold=percentage_bikes_sold).generate()
    renowacje_df, uslugi_lista_ren = RenowacjeDB(renowacje_pojazdy_df['czy_sprzedane']).generate()
    kupno_sprzedaz_df = Kupno_SprzedazDB(klienci_df.index, renowacje_pojazdy_df, renowacje_df).generate()
    
    naprawy_pojazdy_df = Naprawy_PojazdyDB(cars_size=size-cars_size_reno, bikes_size=size-bikes_size_reno).generate()
    naprawy_df, uslugi_lista = NaprawyDB(klienci_df.index, naprawy_pojazdy_df.index).generate()
    
    uslugi_napraw_df = Uslugi_NaprawDB(uslugi_lista, naprawy_df.index, pracownicy_df).generate()
    uslugi_renowacji_df = Uslugi_RenowacjiDB(uslugi_lista_ren, renowacje_df.index, pracownicy_df).generate()

    return (kraje_df, miejscowosci_df, adresy_df, klienci_df, pracownicy_df,
            stanowiska_df, czesci_df, uslugi_df, uslugi_czesci_df,
            wyposazenie_df, renowacje_pojazdy_df, renowacje_df, naprawy_df, uslugi_napraw_df, 
            uslugi_renowacji_df, kupno_sprzedaz_df, naprawy_pojazdy_df)

# Moze niepotrzebne
def drop_tables(engine):
    connection = engine.connect()
    transaction = connection.begin()
    try:
        connection.execute("SET FOREIGN_KEY_CHECKS=0;")
        for table in reversed(Base.metadata.sorted_tables):
            connection.execute(f"DROP TABLE IF EXISTS {table.name};")
        connection.execute("SET FOREIGN_KEY_CHECKS=1;")
        transaction.commit()
    except:
        transaction.rollback()
        raise
    finally:
        connection.close()


def init_db():
    # drop_tables(engine)
    Base.metadata.drop_all(bind=engine)  # Drop all tables to start fresh
    print("All tables dropped successfully.")
    Base.metadata.create_all(bind=engine)  # Create all tables
    print("All tables created successfully.")


def fill_database():
    init_db()
    db = SessionLocal()

    print('Generating data...')
    (kraje_df, miejscowosci_df, adresy_df, klienci_df, pracownicy_df,
     stanowiska_df, czesci_db, uslugi_df, uslugi_czesci_df, wyposazenie_df,
     renowacje_pojazdy_df, renowacje_df, naprawy_df, uslugi_napraw_df, 
     uslugi_renowacji_df, kupno_sprzedaz_df, naprawy_pojazdy_df) = load_data(size=1758, 
                                                                             employees_size=18, 
                                                                             cars_size_reno=157, 
                                                                             bikes_size_reno=101,
                                                                             percentage_cars_sold=0.95,
                                                                             percentage_bikes_sold=0.95)
    print('Data generated successfully :)')
    
    try:
        print("Filling the database with data...")
        
        for i, row in kraje_df.iterrows():
            kraj = Kraje(nazwa=row['nazwa'])
            db.add(kraj)

        for i, row in miejscowosci_df.iterrows():
            miejscowosc = Miejscowosci(
                nazwa=row['nazwa'], 
                id_kraju=row['id_kraju'])
            db.add(miejscowosc)

        for i, row in adresy_df.iterrows():
            adres = Adresy(
                ulica=row['ulica'],
                nr_budynku=row['nr_budynku'],
                nr_mieszkania=row['nr_mieszkania'],
                kod_pocztowy=row['kod_pocztowy'],
                id_miejscowosci=row['id_miejscowosci']
            )
            db.add(adres)

        for i, row in klienci_df.iterrows():
            klient = Klienci(
                imie=row['imie'],
                nazwisko=row['nazwisko'],
                data_urodzenia=row['data_urodzenia'],
                nr_telefonu=row['nr_telefonu'],
                email=row['email'],
                id_adresu=row['id_adresu']
            )
            db.add(klient)

        for i, row in stanowiska_df.iterrows():
            stanowisko = Stanowiska(
                nazwa=row['Nazwa'],
                pensja_min=row['Pensja_min'],
                pensja_max=row['Pensja_max']
            )
            db.add(stanowisko)
            
        for i, row in pracownicy_df.iterrows():
            pracownik = Pracownicy(
                imie = row['imie'],
                nazwisko=row['nazwisko'],
                data_urodzenia=row['data_urodzenia'],
                nr_telefonu=row['nr_telefonu'],
                email=row['email'],
                id_adresu=row['id_adresu'],
                id_stanowiska=row['id_stanowiska'],
                pensja=row['pensja']
            )
            db.add(pracownik)

        for i, row in czesci_db.iterrows():
            czesc = Czesci(
                nazwa_czesci=row['Nazwa czesci'],
                cena=row['ceny za sztuke']
            )
            db.add(czesc)

        for i, row in uslugi_df.iterrows():
            usluga = Uslugi(
                nazwa_uslugi=row['Nazwa usługi'],
                typ_uslugi=row['Typ usługi'],
                naprawa_renowacja=row['Naprawa/Renowacja'],
                stanowisko=row['Stanowisko'],
                cena_dolna=row['Cena dolna'],
                cena_gorna=row['Cena gorna']
            )
            db.add(usluga)

        for index, row in uslugi_czesci_df.iterrows():

            usluga_czesc = Uslugi_Czesci(
                id_uslugi=row['ID_Uslugi'],
                id_czesci=row['ID_Czesci']
            )
            db.add(usluga_czesc)
            
        for i, row in wyposazenie_df.iterrows():
            wyposazenie = Wyposazenie(
                nazwa=row['nazwa'],
                typ_przeznaczenia=row['typ_przeznaczenia'],
                cena=row['cena']
            )
            db.add(wyposazenie)

        for i, row in renowacje_pojazdy_df.iterrows():
            vehicle = Renowacje_Pojazdy(
                typ_pojazdu=row['typ_pojazdu'],
                marka=row['marka'],
                model=row['model'],
                rok_produkcji=row['rok_produkcji'],
                konie_mechaniczne=row['konie_mechaniczne'],
                czy_sprzedane=row['czy_sprzedane'],
                stan=row['stan'],
                naped=row['naped'],
                rodzaj_paliwa=row['rodzaj_paliwa'],
                skrzynia_biegow=row['skrzynia_biegow'],
                przebieg=row['przebieg']
            )
            db.add(vehicle)
        
        for i, row in naprawy_pojazdy_df.iterrows():
            vehicle = Naprawy_Pojazdy(
                typ_pojazdu=row['typ_pojazdu'],
                marka=row['marka'],
                model=row['model'],
                rok_produkcji=row['rok_produkcji'],
                stan=row['stan'],
                naped=row['naped'],
                rodzaj_paliwa=row['rodzaj_paliwa'],
                skrzynia_biegow=row['skrzynia_biegow'],
                przebieg=row['przebieg']
            )
            db.add(vehicle)
            
        for i, row in naprawy_df.iterrows():
            repair = Naprawy(
                id_klienta=row['id_klienta'],
                id_pojazdu=row['id_pojazdu'],
                data_rozpoczecia=row['data_rozpoczecia'],
                data_zakonczenia=row['data_zakonczenia']
            )
            db.add(repair)
            
        for i, row in uslugi_napraw_df.iterrows():
            wykonana_usluga = Uslugi_Napraw(
                id_uslugi=row['id_uslugi'],
                koszt_uslugi=row['koszt_uslugi'],
                id_naprawy=row['id_naprawy'],
                id_pracownika=row['id_pracownika']
            )
            db.add(wykonana_usluga)

        for i, row in renowacje_df.iterrows():
            renovation = Renowacje(
                id_pojazdu=row['id_pojazdu'],
                data_rozpoczecia=row['data_rozpoczecia'],
                data_zakonczenia=row['data_zakonczenia']
            )
            db.add(renovation)

        for i, row in uslugi_renowacji_df.iterrows():
            wykonana_renowacja = Uslugi_Renowacji(
                id_uslugi=row['id_uslugi'],
                koszt_uslugi=row['koszt_uslugi'],
                id_renowacji=row['id_renowacji'],
                id_pracownika=row['id_pracownika']
            )
            db.add(wykonana_renowacja)
        
        for i, row in kupno_sprzedaz_df.iterrows():
            kupno_sprzedaz = Kupno_Sprzedaz(
                typ=row['typ'],
                id_klienta=row['id_klienta'],
                id_pojazdu=row['id_pojazdu'],
                kwota=row['kwota'],
                data=row['data']
            )
            db.add(kupno_sprzedaz)
            
        print('Data added successfully.')
        print("Committing the data to the database...")
        db.commit()
        print("Data committed successfully :)")
    except Exception as e:
        db.rollback()
        print("Error occurred :( ", e)
    finally:
        db.close()
        print("Database commit session closed.")


if __name__ == "__main__":
    fill_database()
