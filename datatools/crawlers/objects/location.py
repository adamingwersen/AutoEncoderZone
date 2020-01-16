from sqlalchemy import Table, Column, Integer, DateTime, String, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

from datetime import datetime
from pprint import pprint
import hashlib

from dbtools.base import Base, session_factory
import objects.house as HouseObj

# Declare base for engine metadata


class Location(Base):
    """
    A table-object which is to be stored:
        Base        : The SQLAlchemy Interface 
        __init__    : Enable initalization with LocationItem Object  
    """
    __tablename__   = 'location'
    id              = Column(Integer, primary_key=True)
    house_id        = Column(Integer, ForeignKey("houses.id"), nullable = False)
    id_hash         = Column(String)
    house           = relationship(HouseObj.House, primaryjoin=house_id==HouseObj.House.id, backref = "locations")
    collected_date  = Column(DateTime)
    house_url       = Column(String)
    loc_lon         = Column(Float)
    loc_lat         = Column(Float)

    def __init__(self, obj):
        if isinstance(obj, LocationObject):
            self.collected_date = datetime.now()
            self.house_id = obj.house_id
            self.house_url = obj.house_url
            self.id_hash = obj.id_hash
            self.loc_lon = obj.loc_lon
            self.loc_lat = obj.loc_lat
        else:
            raise "The provided object is not of type Location(Base)"


class LocationObject(object):
    """
    A class for filling in values from crawler
        __init__    : Initialize object with a url
        hash_url()  : Hashes the provided url for identification of House/Location
        printobj()  : Print the contents of the object, json-like
    """
    def __init__(self, house_url, house_id):
        self.house_url = house_url
        self.house_id = house_id
        self.id_hash = self.hash_url()
        self.loc_lon = float
        self.loc_lat = float

    def hash_url(self):
        return(hashlib.md5(self.house_url.encode()).hexdigest())

    def printobj(self):
        pprint(vars(self))

def insert_db(engine, image):
    session = session_factory()

    to_db = Location(image)
    session._model_changes = {}
    session.add(to_db)
    session.commit()