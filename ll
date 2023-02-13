import pandas as pd
from surprise import SVD, Dataset, Reader

# Load dataset
data = pd.read_csv('customer_purchases.csv')

# Convert data to Surprise format
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(data[['memberid', 'product', 'date']], reader)

# Split data into training and test sets
train_data = data.build_full_trainset()
test_data = train_data.build_anti_testset()

# Train a SVD model on the training data
model = SVD()
model.fit(train_data)

# Make a prediction for the test data
predictions = model.test(test_data)

# Group predictions by user ID and sort by predicted rating
user_predictions = {}
for uid, iid, true_r, est, _ in predictions:
    if uid not in user_predictions:
        user_predictions[uid] = []
    user_predictions[uid].append((iid, est))
for uid, ratings in user_predictions.items():
    ratings.sort(key=lambda x: x[1], reverse=True)

# Print the top predicted items for a given user
user_id = '1234'
top_n = 5
top_items = [iid for (iid, _) in user_predictions[user_id][:top_n]]
print(top_items)
