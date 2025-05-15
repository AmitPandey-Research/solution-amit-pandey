import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('../data/raw/cars_dataset.csv')

# Calculate the variance of NCAP Global Rating
variance = df['NCAP Global Rating'].var()

# Plot the variance
plt.figure()
plt.bar('Variance', variance)
plt.xlabel('Variance of NCAP Global Rating')
plt.ylabel('Value')
plt.title('Variance of NCAP Rating of All Cars')
plt.savefig('variance_of_NCAP_rating.png')
plt.show()