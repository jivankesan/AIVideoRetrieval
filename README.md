# Video Retrieval Application

This is a Flask-based web application that allows users to upload videos, process them, search for similar videos, view all uploaded videos, and manage video entries. The application also includes a dashboard that displays statistics about the actions detected in the uploaded videos.

## Features

- **Upload Videos**: Users can upload videos to the server.
- **Search Videos**: Users can search for similar videos based on text queries.
- **View All Videos**: Users can view, play, and delete uploaded videos.
- **Dashboard**: Users can view a summary of actions detected across all videos.
- **Manage Database**: Users can delete individual video entries or reset the entire database.

## Getting Started

### Prerequisites

- **Python 3.10+**
- **Git** (for cloning the repository)

### Installation

1. **Clone the Repository**

   ```sh
   git clone https://github.com/your-username/video-retrieval-app.git
   cd video-retrieval-app
   ```

2. **Create a Virtual Environment**

   Create a virtual environment to isolate your project's dependencies.

   ```sh
   python -m venv venv
   ```

3. **Activate the Virtual Environment**

   - On **macOS/Linux**:
     ```sh
     source venv/bin/activate
     ```
   - On **Windows**:
     ```sh
     venv\Scripts\activate
     ```

4. **Install the Dependencies**

   Install the required packages using `pip`:

   ```sh
   pip install -r requirements.txt
   ```

5. **Set Up the Database**

   Initialize the database using Flask-Migrate:

   ```sh
   flask db init
   flask db migrate -m "Initial migration"
   flask db upgrade
   ```

6. **Run the Application**

   Start the Flask development server:

   ```sh
   flask run
   ```

7. **Access the Application**

   Open your web browser and go to:

   ```
   http://127.0.0.1:5000/
   ```

## Usage

### Upload a Video

- Click on the "Upload Video" button on the homepage.
- Select a video file to upload.

### Search Videos

- Click on the "Search Videos" button on the homepage.
- Enter a text query to search for similar videos.

### View All Videos

- Click on the "View All Videos" button on the homepage.
- Scroll through the list of uploaded videos, play them, or delete them.

### View Dashboard

- Click on the "View Dashboard" button on the homepage.
- View a summary of the actions detected across all videos.

### Manage Database

- From the "View All Videos" page, delete individual videos.
- From the "Dashboard" page, reset the entire database.

## Resetting the Database

To reset the database from the terminal:

1. **Open Flask Shell**:

   ```sh
   flask shell
   ```

2. **Reset the Database**:

   ```python
   from app import db
   db.drop_all()
   db.create_all()
   ```

3. **Exit Flask Shell**:

   ```python
   exit()
   ```

## Contributing

If you wish to contribute to this project, please fork the repository and create a pull request with your changes. Ensure that your code follows the project's coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any issues or questions, please open an issue on the repository or contact the repository owner.

---

This `README.md` file provides step-by-step instructions for cloning the repository, setting up the environment, and running the Flask application. It also includes basic usage instructions, information on resetting the database, and contributing guidelines. You can modify this file to suit your specific project requirements.
