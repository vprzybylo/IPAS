import ipas
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import Engine
from sqlalchemy import create_engine, event, select
from sqlite3 import Connection as SQLite3Connection
import time
moment=time.strftime("%Y-%b-%d__%H",time.localtime())
f = 'sqlite:///IPAS_'+moment+'.sqlite'

@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    if isinstance(dbapi_connection, SQLite3Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON;")
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.close()
        
engine = create_engine(f)
event.listen(engine, 'connect', _set_sqlite_pragma)   
ipas.Base.metadata.create_all(engine, checkfirst=True)
Session = sessionmaker(bind=engine)
