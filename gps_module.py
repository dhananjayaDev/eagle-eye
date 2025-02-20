import serial
import pynmea2
import random

# Try to connect to the GPS module
try:
    gps_serial = serial.Serial('/dev/ttyUSB0', baudrate=9600, timeout=1)
    gps_available = True
except serial.SerialException:
    gps_available = False  # No GPS module found

def get_gps_data():
    """Fetches GPS data from the module or returns simulated data if GPS is unavailable."""
    if gps_available:
        while True:
            try:
                line = gps_serial.readline().decode('utf-8', errors='ignore')
                if line.startswith("$GPRMC"):
                    msg = pynmea2.parse(line)
                    latitude = float(msg.latitude) if msg.latitude else 0
                    longitude = float(msg.longitude) if msg.longitude else 0
                    speed_knots = float(msg.spd_over_grnd) if msg.spd_over_grnd else 0
                    speed_kmh = speed_knots * 1.852  # Convert knots to km/h
                    return {"latitude": latitude, "longitude": longitude, "speed": round(speed_kmh, 2)}
            except Exception as e:
                print(f"GPS Read Error: {e}")
                return {"latitude": 0, "longitude": 0, "speed": 0}

    # Simulated GPS data when GPS module is unavailable
    return {
        "latitude": round(random.uniform(-90, 90), 6),
        "longitude": round(random.uniform(-180, 180), 6),
        "speed": round(random.uniform(0, 120), 2)  # Simulate speed between 0-120 km/h
    }
