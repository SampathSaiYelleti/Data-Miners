import pandas as pd;

#Read entire dataset with 350k rows
df=pd.read_excel("C:/Users/sampa/Desktop/dataset-news.xlsx")

#Drop unwanted columns
df=df.drop(['id','page_id', 'caption', 'post_type', 'status_type', 'link', 'picture'],axis=1)

#Creating new column by extracting years
df['year'] = df['posted_at'].dt.year



#Read the filtered data
df1=pd.read_excel("C:/Users/sampa/Desktop/Dataset-Main.xlsx")
df1['year'] = df1['posted_at'].dt.year

df1=df1.drop(['id','page_id', 'caption', 'post_type', 'status_type', 'link', 'picture'],axis=1)

average_likes = df.groupby('source')['likes_count'].mean().reset_index()

# Create a dictionary to map sources to their average likes
source_to_average_likes = dict(zip(average_likes['source'], average_likes['likes_count']))

# Define a function to apply the division
def divide_likes_by_average(row):
    source = row['source']
    # Using 1.0 as a default to avoid division by zero
    return row['likes_count'] / source_to_average_likes.get(source, 1.0)

# Apply the division to df_filtered
df1['likes_count_normalized'] = df1.apply(divide_likes_by_average, axis=1)


average_comments = df.groupby('source')['comments_count'].mean().reset_index()

# Create a dictionary to map sources to their average likes
source_to_average_comments = dict(zip(average_comments['source'], average_comments['comments_count']))

# Define a function to apply the division
def divide_comments_by_average(row):
    source = row['source']
    # Using 1.0 as a default to avoid division by zero
    return row['comments_count'] / source_to_average_comments.get(source, 1.0)

# Apply the division to df_filtered
df1['comments_count_normalized'] = df1.apply(divide_comments_by_average, axis=1)



average_shares = df.groupby('source')['shares_count'].mean().reset_index()

# Create a dictionary to map sources to their average shares
source_to_average_shares = dict(zip(average_shares['source'], average_shares['shares_count']))

# Define a function to apply the division
def divide_shares_by_average(row):
    source = row['source']
    # Using 1.0 as a default to avoid division by zero
    return row['shares_count'] / source_to_average_shares.get(source, 1.0)

# Apply the division to df1
df1['shares_count_normalized'] = df1.apply(divide_shares_by_average, axis=1)


# Round 'likes_count_normalized', 'comments_count_normalized',
# and 'shares_count_normalized' to 3 decimal places
df1['likes_count_normalized'] = df1['likes_count_normalized'].round(3)
df1['comments_count_normalized'] = df1['comments_count_normalized'].round(3)
df1['shares_count_normalized'] = df1['shares_count_normalized'].round(3)






print("-----------------------------------------------------------------------------------------------")

#Sort by event code and posted time
df_sorted = df1.sort_values(by=['Event Code', 'posted_at'])

#Rank by posted time and likes count
#Best rank for earliest post time
#Best rank for most liked
df_sorted['Rank_Time'] = df_sorted.groupby('Event Code')['posted_at'].rank(ascending=True).astype(int)
df_sorted['Rank_Likes'] = df_sorted.groupby('Event Code')['likes_count'].rank(ascending=False).astype(int)

print(df_sorted[["Event Code", "posted_at","Rank_Time","likes_count","Rank_Likes"]])
print("-----------------------------------------------------------------------------------------------")
df_sorted['Min Rank'] = df_sorted.groupby('Event Code')['Rank_Time'].transform('min')
df_sorted['Max Rank'] = df_sorted.groupby('Event Code')['Rank_Time'].transform('max')

# Scale the ranks for each event between 0 and 1
df_sorted['Scaled Rank'] = (df_sorted['Rank_Time'] - df_sorted['Min Rank'])/ (df_sorted['Max Rank']-df_sorted['Min Rank'])
df1['Scaled Rank'] = (df_sorted['Rank_Time'] - df_sorted['Min Rank'])/ (df_sorted['Max Rank']-df_sorted['Min Rank'])
df1['Scaled Rank']=df1['Scaled Rank'].round(3)
print(df_sorted)



df_source_rank_time=df_sorted.groupby(['Category','source'])['Scaled Rank'].mean().reset_index()
print(df_source_rank_time)



print("-----------------------------------------------------------------------------------------------")
df_source_rank_likes=df_sorted.groupby(['Category','source'])['likes_count'].mean().reset_index()
print(df_source_rank_likes)
print("-----------------------------------------------------------------------------------------------")



df_metrics=pd.DataFrame()
#df_metrics=df1.groupby(['source'])['likes_count'].mean().reset_index()
#df_metrics=df1.groupby(['source'])['comments_count'].mean().reset_index()
df_metrics = df.groupby(['year','source'])[['likes_count', 'comments_count']].mean().reset_index()
print(df_metrics)
print("-----------------------------------------------------------------------------------------------")



import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.bar(df_metrics['source'], df_metrics['likes_count'], color='skyblue')
ax1.set_xlabel('Source')
ax1.set_ylabel('Average Likes Count')
ax1.set_title('Average Likes Count by Source')
ax1.set_xticklabels(df_metrics['source'], rotation=45, ha='right')

ax2.bar(df_metrics['source'], df_metrics['comments_count'], color='lightcoral')
ax2.set_xlabel('Source')
ax2.set_ylabel('Average Comments Count')
ax2.set_title('Average Comments Count by Source')
ax2.set_xticklabels(df_metrics['source'], rotation=45, ha='right')

plt.tight_layout()
plt.show()



from textblob import TextBlob

df1['Combined_Sentiment_textblob'] = df1['name'].astype(str).apply(lambda text: TextBlob(text).sentiment.polarity)

# Classify sentiment labels based on the polarity
df1['Combined_Sentiment_Label_textblob'] = df1['Combined_Sentiment_textblob'].apply(lambda polarity: 'Positive' if polarity > 0else ('Negative' if polarity < 0 else 'Neutral'))



print(df1.head())




import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()


df1['Combined_Sentiment_nltk'] = df1['name'].astype(str).apply(lambda text: sia.polarity_scores(text)['compound'])

df_sorted['Combined_Sentiment_nltk'] = df_sorted['name'].astype(str).apply(lambda text: sia.polarity_scores(text)['compound'])

df_sorted[["likes_count","comments_count","shares_count","Scaled Rank","Combined_Sentiment_nltk","source"]].to_csv("C:/Users/sampa/Desktop/test_df.csv")


# Classify sentiment labels based on the compound score
df1['Combined_Sentiment_Label_nltk'] = df1['Combined_Sentiment_nltk'].apply(lambda score: 'Positive' if score > 0 else ('Negative' if score < 0 else 'Neutral'))

#print(df1[["name","Combined_Sentiment_Label_textblob","Combined_Sentiment_Label_nltk"]])
print(df1.columns)





df1[["likes_count_normalized","comments_count_normalized","shares_count_normalized","Combined_Sentiment_nltk","Scaled Rank","source"]].to_csv("C:/Users/sampa/Desktop/output_main.csv")

df1[["name", "Combined_Sentiment_Label_textblob","Combined_Sentiment_Label_nltk"]].to_csv("C:/Users/sampa/Desktop/outputdf1.csv")


'''
Need to write code to compare accuracies of both sentiment analyzers 

'''


#Code for visualizing likes count by year for each news source
import seaborn as sns
sns.set(style="darkgrid")

plt.figure(figsize=(10, 6))  # Set the figure size


sns.lineplot(x='year', y='likes_count', hue='source',ci=None,data=df)


plt.xlabel('Year')
plt.ylabel('Likes Count')
plt.title('Likes Count Over Time by Source')


plt.legend(title='Source')
plt.show()


from wordcloud import WordCloud

stopwords = set([ "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up",
    "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once","nan","timeline","photos","-","/","&"])

text = ' '.join(df['name'].astype(str))

# Tokenize the text
words = text.split()  # Split text into words

# Filter out stopwords and convert words to lowercase
filtered_words = [word.lower() for word in words if word.lower() not in stopwords]

# Generate the word cloud
filtered_text = ' '.join(filtered_words)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(filtered_text)

# Display the word cloud using Matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

unique_sources = df['source'].unique()
# Create word clouds for each unique 'source'
for source in unique_sources:
    # Filter the DataFrame for the specific source
    source_df = df[df['source'] == source]

    # Combine the 'name' values for the current source
    text = ' '.join(source_df['name'].astype(str))
    # Tokenize the text
    words = text.split()
    # Filter out stopwords and convert words to lowercase
    filtered_words = [word.lower() for word in words if word.lower() not in stopwords]
    # Generate the word cloud
    filtered_text = ' '.join(filtered_words)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(filtered_text)
    # Display the word cloud with the source name as the title
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for Source: {source}')
    plt.show()

from collections import Counter

word_frequencies = {}

# Iterate through unique sources
unique_sources = df['source'].unique()
for source in unique_sources:
    source_df = df[df['source'] == source]
    text = ' '.join(source_df['name'].astype(str))

    # Tokenize the text and filter out stopwords
    words = [word.lower() for word in text.split() if word.lower() not in stopwords]

    # Count word frequencies using Counter
    word_counts = Counter(words)

    # Get the top 10 words
    top_words = word_counts.most_common(10)

    # Store the top words for this source in the dictionary
    word_frequencies[source] = top_words

# Plot column graphs for each source
for source, top_words in word_frequencies.items():
    words, frequencies = zip(*top_words)  # Unzip words and frequencies
    plt.figure(figsize=(10, 5))
    plt.bar(words, frequencies)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title(f'Top 10 Words for Source: {source}')
    plt.xticks(rotation=45)
    plt.show()



