import schedule
import os
import time
import logging
import traceback
from threading import Thread
from training.training import process_data_and_save
from data_collection.data_collection_2 import collect_data_and_save
from app import app

import os

# Get the directory of the current script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Create a directory for logs relative to the script directory
log_directory = os.path.join(script_directory, 'logs')

if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# Configure logging
logging.basicConfig(
    filename=os.path.join(log_directory, "script_log.txt"),
    level=logging.INFO,  # Set the logging level as needed
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def run_app():
    app.run(debug=False)

def update_app():
    logging.info("Updating the Flask app...")  # Add your update logic here

def main():
    # Start a separate thread to run the Flask app
    app_thread = Thread(target=run_app)
    app_thread.daemon = True
    app_thread.start()

    while True:
        # Fetch data and save every 5 minutes
        collect_data_and_save()

        # Update the model every 5 minutes
        process_data_and_save()

        # Update the Flask app every 60 minutes
        update_app()

        time.sleep(300)  # Sleep for 60 minutes (3600 seconds)

if __name__ == "__main__":
    main()