import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
print(plt.style.available)

# Set style for better visualizations
try:
    plt.style.use('seaborn')
except:
    plt.style.use('ggplot')

sns.set_palette("husl")

# Read the data
data = pd.read_csv('data/ner_train.csv')

# Data Quality Checks
def print_data_quality_report(df):
    print("\n=== Data Quality Report ===")
    print(f"Total rows: {len(df)}")
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    
print_data_quality_report(data)

# Basic Statistics
print("\n=== Basic Statistics ===")
print(data.describe())

# Create subplots for multiple visualizations
fig = plt.figure(figsize=(20, 15))

# 1. Sentence Length Distribution
plt.subplot(2, 2, 1)
sentence_lengths = data.groupby('Sentence #').size()
sns.histplot(sentence_lengths, bins=30, color='skyblue')
plt.title('Sentence Length Distribution')
plt.xlabel('Words per Sentence')
plt.ylabel('Frequency')

# 2. Tag Distribution (excluding 'O' Tag for better visualization)
plt.subplot(2, 2, 2)
tag_counts = data['Tag'].value_counts()
tag_counts_no_O = tag_counts[tag_counts.index != 'O']
sns.barplot(x=tag_counts_no_O.index, y=tag_counts_no_O.values)
plt.title('NER Tag Distribution (excluding O Tag)')
plt.xticks(rotation=45)
plt.xlabel('NER Tags')
plt.ylabel('Frequency')

# 3. Word Length Distribution
plt.subplot(2, 2, 3)
word_lengths = data['word'].str.len()
sns.histplot(word_lengths, bins=30, color='lightgreen')
plt.title('Word Length Distribution')
plt.xlabel('Characters per Word')
plt.ylabel('Frequency')

# 4. Tag Transition Heatmap
plt.subplot(2, 2, 4)
# Create Tag transitions
tag_transitions = []
for _, group in data.groupby('Sentence #'):
    tags = group['Tag'].tolist()
    for i in range(len(tags)-1):
        tag_transitions.append((tags[i], tags[i+1]))

transition_matrix = pd.DataFrame(Counter(tag_transitions).items())
if not transition_matrix.empty:
    transition_matrix = pd.pivot_table(
        transition_matrix,
        values=1,
        index=0,
        columns=1,
        fill_value=0
    )
    sns.heatmap(transition_matrix, cmap='YlOrRd')
    plt.title('Tag Transition Heatmap')

plt.tight_layout()
plt.savefig('ner_analysis.png')
plt.close()

# Additional Statistics
print("\n=== NER Statistics ===")
print("\nTag Distribution:")
print(data['Tag'].value_counts())

print("\nMost Common Words per Tag:")
for Tag in data['Tag'].unique():
    words = data[data['Tag'] == Tag]['word'].value_counts().head(5)
    print(f"\nTop 5 words for {Tag}:")
    print(words)

# Calculate class imbalance
total_tags = len(data)
tag_percentages = (data['Tag'].value_counts() / total_tags * 100).round(2)
print("\nClass Distribution (%):")
print(tag_percentages)
