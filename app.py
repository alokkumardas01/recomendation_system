import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from flask import Flask, request, jsonify
# Call the function to process data and save it
# from training.training import process_data_and_save



app = Flask(__name__)

# Get the directory of the current script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Create a directory for logs relative to the script directory
pickle_directory = os.path.join(script_directory, 'training')
tfidf_vectorizer_pickle_directory=os.path.join(script_directory, 'tfidf_vectorizer.pkl')
purchase_history_pickle_directory=os.path.join(script_directory, 'purchase_history.pkl')


# Load tfidf_matrix from the pickle file
with open(tfidf_vectorizer_pickle_directory, 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Load purchase_history from the pickle file
with open(purchase_history_pickle_directory, 'rb') as file:
    purchase_history = pickle.load(file)

# Define a function to find related customers based on purchase history
def find_related_customers(product_name):
    # Create a pivot table to represent user-item interactions
    pivot_table = purchase_history.pivot_table(index='customerEmailId', columns='product_name', values='purchase_count', fill_value=0)

    # Get the correlation of the product with other products in the pivot table
    correlations = pivot_table.corrwith(pivot_table[product_name])

    # Drop NaN values and sort in descending order
    correlations = correlations.dropna().sort_values(ascending=False)

    return correlations

# Define a function to extract product names from the input data
def extract_product_names(input_data):
    if "data" in input_data and "searchInfo" in input_data["data"]:
        product_list = [item["product_name"] for item in input_data["data"]["searchInfo"]]
        return product_list
    return []

# Define a function to parse the date and time string into a datetime object
def parse_datetime(datetime_str):
    return datetime.strptime(datetime_str, "%A, %B %d, %Y at %I:%M:%S %p")

# Define a function to find related products for a given customer's email
def find_related_products_for_email(customer_email,product_list,target_product_name):
    # Create an empty set to store related products
    related_products_set = set()

    # Iterate through the products purchased by the customer
    for product_name in product_list:
        if product_name != target_product_name:  # Exclude the target product itself
            # Get the correlations for the current product
            correlations = find_related_customers(product_name)

            # Remove the customer's own product from the correlations
            correlations = correlations.drop(customer_email, errors='ignore')

            # Get the top related products for the current product (excluding the target product)
            top_related_products = correlations.head(10).index.tolist()

            # Extend the set of related products
            related_products_set.update(top_related_products)

    # Convert the set to a list and take unique products
    related_products_list = list(related_products_set)

    return related_products_list

# Define a function to find the most similar products to a given product name using cosine similarity
def find_most_similar_products(related_products_list,target_product_name, top_n=5):
    # Calculate the TF-IDF vector for the target product name
    target_product_vector = tfidf_vectorizer.transform([target_product_name])

    # Initialize an empty list to store cosine similarity scores
    similarity_scores = []

    # Iterate through the related products and calculate cosine similarity
    for product in related_products_list:
        product_vector = tfidf_vectorizer.transform([product])
        similarity_score = cosine_similarity(target_product_vector, product_vector)
        similarity_scores.append((product, similarity_score))

    # Sort the list by similarity score in descending order
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # Get the top N most similar products (excluding the target product)
    top_similar_products = [product[0] for product in similarity_scores[:top_n]]

    return top_similar_products


# Define the API endpoint for recommendations
@app.route('/get_recommendations', methods=['GET'])
def get_recommendations():
    # Parse the input JSON data from the request
    input_data = request.json
    # product_list = extract_product_names(input_data)
    customer_email_to_find_related_products = input_data["data"]["customerEmailId"]

    # Access the "searchInfo" list from the JSON data
    search_info = input_data["data"]["searchInfo"]

    # Sort the list of product information dictionaries by "searchDate" using the parse_datetime function
    sorted_search_info = sorted(search_info, key=lambda x: parse_datetime(x["searchDate"]), reverse=True)

    # Use the latest product name as the target_product_name
    target_product_name = sorted_search_info[0]["product_name"]

    # Extract product names from the input data
    product_list = extract_product_names(input_data)

    related_products = find_related_products_for_email(customer_email_to_find_related_products,product_list,target_product_name)
    top_similar_products = find_most_similar_products(related_products,target_product_name, top_n=10)

    # Return the recommendations as JSON response
    return jsonify(top_similar_products)


if __name__ == '__main__':
    app.run(debug=True)