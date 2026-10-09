"""Real historical daily max wind (km/h) per port from Open-Meteo ERA5 archive, cached to CSV."""
import pandas as pd, requests
from pathlib import Path

PORTS = {  # name: (lat, lon)
    "Nhava Sheva": (18.95, 72.95), "Mundra": (22.74, 69.70), "Chennai": (13.08, 80.30),
    "Rotterdam": (51.95, 4.14), "Singapore": (1.26, 103.80), "Jebel Ali": (25.00, 55.06),
    "Los Angeles": (33.73, -118.26), "Hamburg": (53.54, 9.97),
}
CACHE = Path(__file__).resolve().parents[2] / "data" / "wind_daily.csv"
START, END = "2023-01-01", "2024-12-31"


def load_wind() -> pd.DataFrame:
    """Return DataFrame indexed by date, one column per port (wind km/h)."""
    if CACHE.exists():
        return pd.read_csv(CACHE, index_col=0, parse_dates=True)
    cols = {}
    for p, (lat, lon) in PORTS.items():
        r = requests.get("https://archive-api.open-meteo.com/v1/archive", timeout=60, params=dict(
            latitude=lat, longitude=lon, start_date=START, end_date=END,
            daily="wind_speed_10m_max", timezone="UTC"))
        r.raise_for_status()
        d = r.json()["daily"]
        cols[p] = pd.Series(d["wind_speed_10m_max"], index=pd.to_datetime(d["time"]))
    df = pd.DataFrame(cols)
    CACHE.parent.mkdir(exist_ok=True)
    df.to_csv(CACHE)
    return df


if __name__ == "__main__":
    print(load_wind().describe().T)
