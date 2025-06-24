import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
df = pd.read_csv('titanic.csv')

# 1. Generate summary statistics
print("\n=== Summary Statistics ===")
print(df.describe(include='all'))

# 2. Histograms for numeric features
df.hist(figsize=(10,8), bins=20)
plt.suptitle('Histograms')
plt.tight_layout()
plt.savefig('histograms.png')
plt.close()

# 3. Boxplots for numeric features
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(10, 6))
df[num_cols].plot(kind='box')
plt.title('Boxplots')
plt.savefig('boxplots.png')
plt.close()

# 4. Correlation Matrix
corr = df.corr(numeric_only=True)
print("\n=== Correlation Matrix ===\n", corr)
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.close()

# 5. Pairplot of key numeric features
sns.pairplot(df[num_cols].dropna())  
plt.savefig('pairplot.png')
plt.close()

# 6. Detect skewness in data
skewness = df[num_cols].skew().sort_values(ascending=False)
print("\n=== Skewness of numeric columns ===\n", skewness)

# Done!
print("\n✅ EDA Complete. Check the generated images for visualizations!")

