"""Port zones for AIS call extraction. Centres are APPROXIMATE hand-entered coordinates (not an official list); arrival is
defined as first entry into a circle of radius R km around the centre, so small errors only shift the arrival definition.
Every centre was checked against the data: the densest moored location of vessels declaring that code lies within 5 km of the
centre for all 27 ports (this check caught a swapped DKFRC/DKFDH pair and three centres that were 5-10 km off)."""
import re

PORTS = {  # UN/LOCODE: (lat, lon). Only ports inside DMA receiver coverage can be observed.
    "SEGOT": (57.70, 11.88), "DKSKA": (57.72, 10.59), "DERSK": (54.15, 12.10), "DKEBJ": (55.47, 8.43), "DKCPH": (55.70, 12.60),
    "DKAAR": (56.15, 10.23), "DEKEL": (54.33, 10.15), "SEHEL": (56.04, 12.69), "DKFDH": (57.44, 10.55), "DKFRC": (55.56, 9.75),
    "DKAAL": (57.05, 10.02), "DKODE": (55.46, 10.53), "DKGED": (54.57, 11.93), "DKRDB": (54.65, 11.35), "SETRG": (55.37, 13.16),
    "SEMMA": (55.61, 13.00), "DKKAL": (55.68, 11.08), "DKHIR": (57.59, 9.96), "DKTHY": (56.70, 8.21), "DKSVE": (55.06, 10.61),
    "DKNYB": (55.31, 10.79), "DKKOG": (55.46, 12.19), "DKGRE": (56.41, 10.93), "DKRNN": (55.10, 14.69), "DETRV": (53.96, 10.87),
    "DELBC": (53.90, 10.70), "DEPUT": (54.50, 11.22)}
ALIAS = {"SKAGEN": "DKSKA", "ESBJERG": "DKEBJ", "ROSTOCK": "DERSK", "DKRSK": "DERSK", "GOTHENBURG": "SEGOT", "GOTEBORG": "SEGOT",
         "COPENHAGEN": "DKCPH", "KOBENHAVN": "DKCPH", "AARHUS": "DKAAR", "AARUS": "DKAAR", "KIEL": "DEKEL", "HELSINGBORG": "SEHEL",
         "FREDERIKSHAVN": "DKFDH", "FREDERICIA": "DKFRC", "AALBORG": "DKAAL", "ODENSE": "DKODE", "GEDSER": "DKGED", "RODBY": "DKRDB",
         "TRELLEBORG": "SETRG", "MALMO": "SEMMA", "KALUNDBORG": "DKKAL", "HIRTSHALS": "DKHIR", "SVENDBORG": "DKSVE",
         "NYBORG": "DKNYB", "TRAVEMUNDE": "DETRV", "LUBECK": "DELBC", "PUTTGARDEN": "DEPUT", "RONNE": "DKRNN", "GRENAA": "DKGRE",
         "KOGE": "DKKOG", "THYBORON": "DKTHY"}


def locode(dest):
    """Free-text AIS destination -> port code in PORTS, or None. Takes the last leg of 'A>B', drops brackets and non-letters."""
    s = re.sub(r"[^A-Z]", "", str(dest).upper().split(">")[-1].split("(")[0])
    s = ALIAS.get(s, s)
    return s if s in PORTS else None
