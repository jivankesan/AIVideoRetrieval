# Video Retrieval Application

This application allows users to upload videos, process them, and search for similar videos using text queries. The application uses Flask for the web interface, SQLAlchemy for database management, and PyTorch for video processing and embeddings.

## Features

- Upload videos and process them for embedding extraction.
- Store video embeddings in a database.
- Search for similar videos using text queries.
- Display the most similar video with its details.

## Installation

1. **Clone the repository:**

2. **Create a virtual environment:**

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Flask application:**

   ```sh
   flask run
   ```

2. **Open your browser and navigate to:**

   ```
   http://127.0.0.1:5000/
   ```

3. **Upload a Video:**

   - Use the upload form to upload a video file.
   - The video will be processed and its embedding will be stored in the database.

4. **Search for Similar Videos:**

   - Use the search form to enter a text query.
   - The application will return the most similar video based on the text query.
