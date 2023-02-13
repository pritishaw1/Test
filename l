import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Load dataset
data = pd.read_csv('customer_purchases.csv')

# Convert member IDs and products to integers
memberid_to_int = {memberid: i for i, memberid in enumerate(data['memberid'].unique())}
product_to_int = {product: i for i, product in enumerate(data['product'].unique())}
data['memberid'] = data['memberid'].map(memberid_to_int)
data['product'] = data['product'].map(product_to_int)

# Create training data
X_train = data[['memberid', 'product']].values
y_train = np.zeros(X_train.shape[0])
y_train[:-1] = X_train[1:, 1]
y_train[-1] = X_train[0, 1]

# Define model architecture
num_members = len(memberid_to_int)
num_products = len(product_to_int)
embedding_dim = 10
model = Sequential([
    Input(shape=(2,)),
    Embedding(num_members, embedding_dim, input_length=2),
    Flatten(),
    Dense(num_products, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train model
es = EarlyStopping(patience=3, restore_best_weights=True)
model.fit(X_train, y_train, epochs=10, callbacks=[es])

# Make predictions for each member
member_ids = data['memberid'].unique()
next_purchases = {}
for member_id in member_ids:
    member_products = data[data['memberid'] == member_id]['product'].values
    if len(member_products) > 0:
        current_product = member_products[-1]
        predicted_product_probs = model.predict(np.array([[member_id, current_product]]))[0]
        predicted_product = np.argmax(predicted_product_probs)
        next_purchases[member_id] = predicted_product

# Convert product IDs back to product names
int_to_product = {i: product for product, i in product_to_int.items()}
next_purchases = {member_id: int_to_product.get(product_id, None) for member_id, product_id in next_purchases.items()}
print(next_purchases)
