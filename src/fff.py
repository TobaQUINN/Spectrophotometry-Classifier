import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv(r"C:\Users\DARA2\Downloads\StudentPerformanceFactors_082342.csv")

print("Lists of column names:", df.columns.tolist())

"""
Will be plotting a dynamic grouped bar chart to compare engagement of the Genders in Extracurricular activities.

"""

# Step 1: Count how many students in each Gender × Extracurricular_Activities group
counts = df.groupby(["Gender", "Extracurricular_Activities"]).size().unstack()

# Step 2: Plot grouped bar chart
x = np.arange(len(counts.index))  # positions for Male/Female
width = 0.35

fig, ax = plt.subplots()

bars1 = ax.bar(x - width/2, counts["Yes"], width, label="Yes")
bars2 = ax.bar(x + width/2, counts["No"], width, label="No")

# Step 3: Add labels and title
ax.set_xlabel("Gender")
ax.set_ylabel("Number of Students")
ax.set_title("Extracurricular Activities Participation by Gender")
ax.set_xticks(x)
ax.set_xticklabels(counts.index)
ax.legend(title="Extracurricular Activities")

plt.show()