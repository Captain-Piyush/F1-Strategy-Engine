import fastf1
import os

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
    
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    
    fastest_lap = session.laps.pick_fastest()
    
    print(f"--- Data Integrity Check ---")
    print(f"Fastest Driver: {fastest_lap['Driver']}")
    print(f"Top Speed: {fastest_lap['SpeedST']} km/h")
    print(f"Tire Compound: {fastest_lap['Compound']}")
    print(f"----------------------------")
    
    return session

if __name__ == "__main__":
    try:
        data = load_session_data(2024, 'Saudi Arabia', 'R')
        print("✅ SUCCESS: Data Ingestion Engine is online and hardware-ready.")
    except Exception as e:
        print(f"❌ ERROR: Ingestion failed. Check connection or year/GP spelling.")
        print(f"Details: {e}")