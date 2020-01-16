# Libraries
from sqlalchemy import Table, Column, Integer, DateTime, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from pprint import pprint
import hashlib

from dbtools.base import Base, session_factory

class House(Base):
    __tablename__   = "houses"
    id              = Column(Integer, primary_key = True, autoincrement = True)
    id_hash         = Column(String)
    collected_date  = Column(DateTime)
    url             = Column(String, unique = True)
    case_id         = Column(Integer)

    price           = Column(Float)
    owner_expense   = Column(Float)
    energy_label    = Column(String)
    house_type      = Column(String)
    living_sqm      = Column(Integer)
    rooms           = Column(Integer)
    build_year      = Column(Integer)
    time_on_market  = Column(Integer)

    address         = Column(String)
    zipcode         = Column(String)
    city            = Column(String)

    def __init__(self, obj):
        if isinstance(obj, HouseObject):
            self.collected_date = datetime.now()
            self.url            = obj.url
            self.case_id        = obj.case_id
            self.price          = obj.price
            self.owner_expense  = obj.owner_expense
            self.energy_label   = obj.energy_label
            self.house_type     = obj.house_type
            self.living_sqm     = obj.living_sqm
            self.rooms          = obj.rooms 
            self.living_sqm     = obj.living_sqm
            self.build_year     = obj.build_year
            self.time_on_market = obj.time_on_market
            self.address        = obj.address
            self.zipcode        = obj.zipcode
            self.city           = obj.city
        else:
            raise "the provided object is not a House"

class HouseObject(object):
    def __init__(self, url):
        self.url            = url
        self.id_hash        = self.hash_url()
        self.case_id        = int
        self.price          = float
        self.owner_expense  = float
        self.energy_label   = str
        self.house_type     = str
        self.living_sqm     = int
        self.rooms          = int
        self.living_sqm     = int
        self.build_year     = int
        self.time_on_market = int
        self.address        = str
        self.zipcode        = str
        self.city           = str

    def __repr__(self):
        return "House with case_id: {} for price: {} @ {}".format(str(self.case_id), str(self.price), self.city)

    def __str__(self):
        return str(pprint(vars(self)))
        

    def hash_url(self):
        return hashlib.md5(self.url.encode()).hexdigest()

def insert_db(engine, house):
    session = session_factory()
    
    to_db = House(house)
    session._model_changes = {}
    session.add(to_db)
    session.commit()


