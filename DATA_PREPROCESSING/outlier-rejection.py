import pandas as pd
import numpy as np
import Orange

# Load data from Orange
data = in_data  # The dataset passed from the previous widget

# Convert Orange data to pandas DataFrame, including the 'Date' column as meta
df = pd.DataFrame(data.X, columns=[attr.name for attr in data.domain.attributes])
meta_df = pd.DataFrame(data.metas, columns=[attr.name for attr in data.domain.metas])

# Combine data and meta dataframes
combined_df = pd.concat([df, meta_df], axis=1)

# Specify the column to check for outliers
feature = 'Low'  # The column name for outlier detection

# Calculate z-scores
combined_df['z_score'] = (combined_df[feature] - combined_df[feature].mean()) / combined_df[feature].std()

# Determine threshold for outliers (e.g., |z| > 3)
threshold = 3  # Adjust the threshold if needed
combined_df['is_outlier'] = np.abs(combined_df['z_score']) > threshold

# Filter out outliers
filtered_df = combined_df[~combined_df['is_outlier']]

# Drop auxiliary columns used for filtering
filtered_df = filtered_df.drop(columns=['z_score', 'is_outlier'])

# Separate the meta column ('Date') and data columns
data_columns = [attr.name for attr in data.domain.attributes]
meta_columns = [attr.name for attr in data.domain.metas]

filtered_data_df = filtered_df[data_columns]
filtered_meta_df = filtered_df[meta_columns]

# Convert the filtered DataFrame back to Orange format
# Create the domain with data attributes and meta attributes
domain = Orange.data.Domain(
    [Orange.data.ContinuousVariable(attr) for attr in data_columns],
    metas=[Orange.data.StringVariable(attr) for attr in meta_columns]
)

# Combine data and meta values
X = filtered_data_df.values
metas = filtered_meta_df.values

# Create the filtered Orange data table
filtered_data = Orange.data.Table(domain, X, metas=metas)

# Save the filtered data to a CSV file (if needed)
# filtered_df.to_csv('filtered_dataset.csv', index=False)

# Output the filtered dataset
out_data = filtered_data
