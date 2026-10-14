"""
Starter code for building a REAL-DATA proof of concept (Week 2, Appendix B).
NOT TESTED: the environment used to write this had no access to these services.
Check each service's current access method and licence, then adapt.
"""
import requests, pandas as pd

# ---- 1. Daily weather from NASA POWER (no key needed) -------------------------------
def nasa_power_daily(lat, lon, start="20230601", end="20231031"):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    p = dict(parameters="T2M_MAX,T2M_MIN,PRECTOTCORR", community="AG",
             longitude=lon, latitude=lat, start=start, end=end, format="JSON")
    data = requests.get(url, params=p, timeout=60).json()["properties"]["parameter"]
    wx = pd.DataFrame(data); wx.index = pd.to_datetime(wx.index)
    return wx.replace(-999, float("nan"))              # POWER marks missing values as -999

def stage_features(wx, plant_date, stage_days=(0, 40, 80, 120)):
    """Aggregate weather by growth stage counted from planting date (label-free features)."""
    out = {}
    for i in range(len(stage_days) - 1):
        a = pd.Timestamp(plant_date) + pd.Timedelta(days=stage_days[i])
        b = pd.Timestamp(plant_date) + pd.Timedelta(days=stage_days[i + 1])
        w = wx.loc[a:b]
        out[f"rain_s{i+1}"] = w.PRECTOTCORR.sum()
        out[f"heat_days_s{i+1}"] = int((w.T2M_MAX > 35).sum())
        out[f"gdd_s{i+1}"] = (((w.T2M_MAX + w.T2M_MIN) / 2 - 10).clip(lower=0)).sum()   # base 10 C (crop-specific!)
    return out

# ---- 2. Sentinel-2 NDVI through Google Earth Engine (needs an Earth Engine account) -
def s2_ndvi_collection(geometry, start, end):
    import ee; ee.Initialize()
    def f(img):
        ok = img.select("SCL").remap([4, 5], [1, 1], 0)        # keep vegetation / bare soil, drop cloud, shadow
        return (img.updateMask(ok).normalizedDifference(["B8", "B4"]).rename("NDVI")
                   .copyProperties(img, ["system:time_start"]))
    return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(geometry)
              .filterDate(start, end).map(f))

# ---- 3. Smooth an NDVI series (Savitzky-Golay) and derive phenology features --------
def ndvi_features(series):
    from scipy.signal import savgol_filter
    s = series.interpolate(limit_direction="both")              # fill cloud gaps in time
    sm = pd.Series(savgol_filter(s.values, window_length=7, polyorder=2), index=s.index)
    return {"ndvi_peak": sm.max(), "ndvi_peak_date": sm.idxmax(), "ndvi_auc": sm.sum()}
