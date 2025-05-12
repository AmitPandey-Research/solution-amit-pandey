import pandas as pd

# Load the dataset
df = pd.read_csv('../data/raw/cars_dataset.csv')

# Calculate the mean NCAP Global Rating
mean_ncap_rating = df['NCAP Global Rating'].mean()
print(mean_ncap_rating)