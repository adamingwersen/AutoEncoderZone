from sqlalchemy import Table, Column, Integer, DateTime, String, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from urllib.parse import urlparse
from os.path import splitext
from datetime import datetime
from pprint import pprint
import hashlib

from dbtools.base import Base, session_factory
import objects.house as HouseObj


class Image(Base):
    __tablename__   = 'image'
    id              = Column(Integer, primary_key=True)
    house_id        = Column(Integer, ForeignKey("houses.id"), nullable = False)
    house           = relationship(HouseObj.House, primaryjoin=house_id==HouseObj.House.id, backref = "images")
    house_url       = Column(String)
    image_url       = Column(String)
    collected_date  = Column(DateTime)
    file_name       = Column(String)
    file_path       = Column(String)
    file_extension  = Column(String)

    def __init__(self, obj):
        if isinstance(obj, ImageObject):
            self.collected_date     = datetime.now()
            self.house_url          = obj.house_url
            self.house_id           = obj.house_id
            self.image_url          = obj.image_url
            self.file_name          = obj.file_name
            self.file_path          = obj.file_path
            self.file_extension     = obj.file_extension
        else:
            raise "The provided object is not Image"


class ImageObject(object):
    def __init__(self, house_url, house_id, image_url):
        self.house_url          = house_url
        self.house_id           = house_id
        self.id_hash            = self.hash_url()
        self.image_url          = image_url
        self.file_path          = str
        self.file_extension     = self.fetch_url_extension()
        self.file_name          = self.gen_filename()

    def fetch_url_extension(self):
        url = urlparse(self.image_url).path
        ext = splitext(url)[1]
        return(ext)

    def hash_url(self):
        return(hashlib.md5(self.house_url.encode()).hexdigest())

    def gen_filename(self): 
        image_file = hashlib.md5(self.image_url.encode()).hexdigest() + self.file_extension
        return(image_file)

    def __repr__(self):
        pprint(vars(self))


def insert_db(engine, image):
    session = session_factory()

    to_db = Image(image)
    session._model_changes = {}
    session.add(to_db)
    session.commit()