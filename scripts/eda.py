import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from collections import defaultdict

# Load the dataset
data = pd.read_csv('data/ner_train.csv')

# Display the first few rows
print("First few rows of the dataset:")
print(data.head())

# Basic Information
num_sentences = data['Sentence #'].nunique()
unique_pos = data['POS'].explode().nunique()
unique_tags = data['Tag'].explode().nunique()

print(f"\nNumber of sentences: {num_sentences}")
print(f"Number of unique POS tags: {unique_pos}")
print(f"Number of unique NER tags: {unique_tags}")

# Distribution of Tags
all_tags = [tag for sublist in data['Tag'].apply(eval) for tag in sublist]
tag_counts = pd.Series(all_tags).value_counts()

plt.figure(figsize=(10, 6))
sns.barplot(x=tag_counts.index, y=tag_counts.values, palette="viridis")
plt.title('Distribution of NER Tags')
plt.xlabel('Tags')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# Sentence Length Analysis
data['Sentence Length'] = data['Sentence'].apply(lambda x: len(x.split()))

plt.figure(figsize=(10, 6))
sns.histplot(data['Sentence Length'], bins=30, kde=True, color='blue')
plt.title('Distribution of Sentence Lengths')
plt.xlabel('Sentence Length')
plt.ylabel('Frequency')
plt.show()

# POS Tag Distribution
all_pos = [pos for sublist in data['POS'].apply(eval) for pos in sublist]
pos_counts = pd.Series(all_pos).value_counts()

plt.figure(figsize=(12, 6))
sns.barplot(x=pos_counts.index, y=pos_counts.values, palette="magma")
plt.title('Distribution of POS Tags')
plt.xlabel('POS Tags')
plt.ylabel('Frequency')
plt.xticks(rotation=90)
plt.show()

# NER Tag Co-occurrence
co_occurrence = defaultdict(int)

for tags in data['Tag'].apply(eval):
    unique_tags_in_sentence = set(tags)
    for tag_pair in combinations(unique_tags_in_sentence, 2):
        co_occurrence[tag_pair] += 1

co_occurrence_df = pd.DataFrame(list(co_occurrence.items()), columns=['Tag Pair', 'Frequency'])
co_occurrence_df = co_occurrence_df.sort_values(by='Frequency', ascending=False)

print("\nTop co-occurring tag pairs:")
print(co_occurrence_df.head(10))

# Entity Analysis
entities = []
for sentence, tags in zip(data['Sentence'], data['Tag'].apply(eval)):
    words = sentence.split()
    for word, tag in zip(words, tags):
        if tag != 'O':
            entities.append((word, tag))

entities_df = pd.DataFrame(entities, columns=['Entity', 'Tag'])
entity_counts = entities_df.groupby(['Entity', 'Tag']).size().reset_index(name='Frequency')

print("\nMost common entities:")
print(entity_counts.sort_values(by='Frequency', ascending=False).head(10))

# Missing Data
print("\nMissing values in the dataset:")
print(data.isnull().sum())

# Visualize Sentences with Entities
def visualize_sentence(sentence, tags):
    words = sentence.split()
    for word, tag in zip(words, tags):
        if tag != 'O':
            print(f"{word} ({tag})", end=' ')
        else:
            print(word, end=' ')
    print()

print("\nVisualizing a few sentences with their entities:")
for i in range(5):
    visualize_sentence(data['Sentence'].iloc[i], eval(data['Tag'].iloc[i]))

# Summary of Findings
print("\nSummary of Findings:")
print(f"- There are {num_sentences} sentences in the dataset.")
print(f"- There are {unique_pos} unique POS tags and {unique_tags} unique NER tags.")
print("- The distribution of NER tags shows the following frequencies:")
print(tag_counts)
print("- The most common entities and their types are:")
print(entity_counts.sort_values(by='Frequency', ascending=False).head(10))
print("- The average sentence length is {:.2f} words.".format(data['Sentence Length'].mean()))
print("- The top co-occurring tag pairs are:")
print(co_occurrence_df.head(10))
print("- There are no missing values in the dataset." if data.isnull().sum().sum() == 0 else "- There are missing values in the dataset.")