from app import db
db.drop_all()  # Deletes the existing database schema
db.create_all()  # Recreates the tables with the new schema
