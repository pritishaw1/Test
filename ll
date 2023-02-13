import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data = pd.read_csv('customer_purchases.csv')

# Split data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2)

# Create a pivot table to represent user-item interactions
user_item_matrix = train_data.pivot_table(index='memberid', columns='product', values='date').fillna(0)

# Compute cosine similarity between user-item matrix
item_similarity = cosine_similarity(user_item_matrix.T)

# Define a function to make recommendations for a given user
def get_recommendations(user_id, top_n):
    user_items = user_item_matrix.loc[user_id,:]
    scores = item_similarity.dot(user_items)
    scores = scores.sort_values(ascending=False)
    top_items = scores.iloc[:top_n].index
    return top_items

# Make a recommendation for a given user
user_id = '1234'
top_n = 5
recommended_items = get_recommendations(user_id, top_n)
print(recommended_items)
