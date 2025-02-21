from flask_sqlalchemy import SQLAlchemy
from flask import Flask
import os
from datetime import datetime

app = Flask(__name__)

# ✅ Configure SQLite Database
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///vehicle_tracking.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


# ✅ Define Database Models
class TrackingData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    vehicle_id = db.Column(db.Integer, nullable=False)
    v_Vel = db.Column(db.Float, nullable=False)
    v_Acc = db.Column(db.Float, nullable=False)
    Lane_ID = db.Column(db.Integer, nullable=False)
    pred_x = db.Column(db.Integer, nullable=False)
    pred_y = db.Column(db.Integer, nullable=False)


class Anomalies(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    vehicle_id = db.Column(db.Integer, nullable=False)
    severity = db.Column(db.String(10), nullable=False)
    anomaly_score = db.Column(db.Float, nullable=False)


# ✅ Initialize Database and Create Tables
if not os.path.exists("../vehicle_tracking.db"):
    with app.app_context():
        db.create_all()
        print("✅ Database and tables created!")


import sqlite3

conn = sqlite3.connect("../vehicle_tracking.db")
cursor = conn.cursor()

# Check if tables exist
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in Database:", tables)

# Verify TrackingData table has records
cursor.execute("SELECT * FROM TrackingData LIMIT 5;")
print("Tracking Data:", cursor.fetchall())

conn.close()

