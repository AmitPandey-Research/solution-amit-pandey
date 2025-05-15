import pandas as pd
import matplotlib.pyplot as plt

# Load the car dataset
df = pd.read_csv('../data/raw/cars_dataset.csv')

# Plot the distribution of NCAP Global Ratings
plt.hist(df['NCAP Global Rating'], bins=5, color='skyblue', edgecolor='black')
plt.xlabel('NCAP Global Rating')
plt.ylabel('Frequency')
plt.title('Distribution of NCAP Global Ratings')
plt.show()