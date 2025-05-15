import pandas as pd

# Load the CSV file
df = pd.read_csv('../data/raw/cars_dataset.csv')

# Calculate the variance of NCAP Global Rating
variance = df['NCAP Global Rating'].var()

variance