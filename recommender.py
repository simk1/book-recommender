import pandas as pd
from surprise import Reader, Dataset, KNNBasic, accuracy
from surprise.model_selection import train_test_split

# Load the dataset
book_ratings = pd.read_csv('goodreads_ratings.csv')

# Print dataset size and examine column data types
print(f"Dataset size: {book_ratings.shape}")
print(book_ratings.dtypes)

# Distribution of ratings
print(book_ratings['rating'].value_counts())

# Filter ratings that are out of range
book_ratings = book_ratings[book_ratings['rating'] != 0]

# Prepare data for surprise: build a Surprise reader object
reader = Reader(rating_scale=(1, 5))

# Load `book_ratings` into a Surprise Dataset
data = Dataset.load_from_df(book_ratings[['user_id', 'book_id', 'rating']], reader)

# Create an 80:20 train-test split and set the random state to 7
trainset, testset = train_test_split(data, test_size=0.2, random_state=7)

# Use KNNBasic from Surprise to train a collaborative filter
recommender = KNNBasic()
recommender.fit(trainset)

# Evaluate the recommender system
predictions = recommender.test(testset)
rmse = accuracy.rmse(predictions)
print(f"RMSE: {rmse}")

# Prediction on a user who gave the "The Three-Body Problem" a rating of 5
# Here, you need to know the `user_id` and `book_id` for "The Three-Body Problem"
# Assuming user_id = 1 and book_id = 1 as an example
user_id = 1
book_id = 1
rating = 5

# Predict the rating
pred = recommender.predict(user_id, book_id, rating)
print(f"Predicted rating for user {user_id} and book {book_id}: {pred.est}")
