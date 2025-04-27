import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset with match scores
df_matched = pd.read_csv(r'D:\cv-matcher\data\matched_UpdatedResumeDataSet.csv')

# Plot a histogram of match scores
plt.figure(figsize=(10, 6))
plt.hist(df_matched['Match_Score'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Match Scores')
plt.xlabel('Match Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
