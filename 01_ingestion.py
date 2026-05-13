import fastf1
import os

# 1. SETUP CACHE 
# This is a mechanical necessity. F1 telemetry is massive. 
# Without a cache, you'll be banned for too many requests.
CACHE_DIR = 'f1_cache'
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

fastf1.Cache.enable_cache(CACHE_DIR)

def load_session_data(year, gp, session_type):
    """
    Retrieves and loads F1 session data.
    'R' = Race, 'Q' = Qualifying.
    """
    print(f"\n>>> Accessing {year} {gp} - {session_type}...")
    
    # get_session points to the event, session.load() actually grabs the data
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    
    # Verification: Get the fastest lap to prove data integrity
    fastest_lap = session.laps.pick_fastest()
    
    print(f"--- Data Integrity Check ---")
    print(f"Fastest Driver: {fastest_lap['Driver']}")
    print(f"Top Speed: {fastest_lap['SpeedST']} km/h")
    print(f"Tire Compound: {fastest_lap['Compound']}")
    print(f"----------------------------")
    
    return session

if __name__ == "__main__":
    # Test with the 2024 Saudi Arabian Grand Prix
    # We use this as a 'Cold Start' test to ensure cache and API are talking.
    try:
        data = load_session_data(2024, 'Saudi Arabia', 'R')
        print("✅ SUCCESS: Data Ingestion Engine is online and hardware-ready.")
    except Exception as e:
        print(f"❌ ERROR: Ingestion failed. Check connection or year/GP spelling.")
        print(f"Details: {e}")