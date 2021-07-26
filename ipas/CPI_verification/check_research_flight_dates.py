import pandas as pd

df = pd.read_csv("final_databases/no_mask/CRYSTAL_FACE_UND_datesorted.csv")
dates = pd.to_datetime(df["date"])
df = df.set_index(["date"])
bad_dates = ["08", "10", "17", "22", "27"]
full_dates = []
for day in bad_dates:
    full_dates.append("2002-07-" + day)
match = df[df.index.to_series().dt.date.astype(str).isin(full_dates)]
print(match)
print(day, len(match))
