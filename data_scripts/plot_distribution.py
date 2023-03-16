import pandas as pd
import matplotlib.pyplot as plt

csv_file = '../data/all_apache_commits.csv'
delimiter = ', '

# Read the CSV file
df = pd.read_csv(csv_file, sep=delimiter)

# Extract the 4th column (assuming 0-based indexing)
fourth_column = df.iloc[:, 3]

# Plot the distribution
plt.hist(fourth_column, bins='auto', edgecolor='black')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution of the 4th Column')

# Save the plot as a PNG image
plt.savefig('distribution_plot.png')

# Show the plot
plt.show()
