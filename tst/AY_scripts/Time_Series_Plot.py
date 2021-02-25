import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

Base_Path = Path("Z:\\comma_group1\\Rom\\dat\\AaronsTraining\\Elec_Price\\Special_Issue\\Training_Times\\Time_Analysis\\Time_Series")
source = "Data-Jan.csv"
df = pd.read_csv(Base_Path / source, header=[0], index_col=0)
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel('Date and Time', fontsize=14)
ax1.set_ylabel('Electricity Price (£/MWh)', fontsize=14)
ax1 = sns.lineplot(data=df, x="Date_Time", y="Observed Price",
                   size=0.5, color="blue", marker="x")
ax2 = ax1.twinx()
plt.show()
