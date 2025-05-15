import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file
df = pd.read_csv('../data/raw/cars_dataset.csv')

# Create a binned histogram of user ratings
plt.hist(df['User Ratings'], bins=10, range=(0, 5), edgecolor='black')
plt.xlabel('User Ratings')
plt.ylabel('Frequency')
plt.title('Binned Histogram of User Ratings of Cars')
plt.savefig('binned_histogram_user_ratings.png')
plt.show()