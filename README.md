# LIfeHack2024
# Patrol Planner Web Application

The Patrol Planner web application utilizes past crime data to suggest optimal patrol routes to police officers based on their patrol area, time of patrol, and available number of patrol vehicles.

## Installation

To install the necessary dependencies, follow these steps:

1. From the root directory, run the following command to install Python dependencies:
   `pip install -r requirements.txt`

3. Navigate to the `/patrolplanner` directory and run the following command to install Node.js dependencies:
 `cd patrolplanner`
 `npm install`


## Starting the Servers

For the app to work, both the backend and frontend server must be running CONCURRENTLY. Start the servers on separate terminals.


To start the backend server, follow these steps:

1. Navigate to the `/backend` directory.
2. Run the following command to start the backend server:
   `python3 app.py`
   
To start the frontend, follow these steps:

1. Navigate to the `/patrolplanner` directory.
2. Run the following command to start the frontend server:
   `npm start`


## Usage

1. Once the application is running, open it in your web browser.
2. Click on the box on the left side of the map to highlight the grid area to patrol.
3. Import a CSV file containing fake crime data with the following headers: `lat`, `long`, `date`, `severity` (where severity ranges from 1 to 10).
4. Click the "Import .csv" button to import the crime data. => **sample csv files have been provided in additional information section of submission**
5. The application will suggest optimal patrol routes based on the imported data and other parameters
6. The relevant data and images will be downloaded into a folder called plot_images in the ROOT DIRECTORY.
7. Open the file plot_images from your file explorer to obtain the relevant images
8. **BUG**: if some images become faulty/distorted after multiple executions of the program, try clearing cache or restarting front and backend servers to solve the issue

## Contributing

If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature (`git checkout -b feature-name`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-name`).
6. Create a new Pull Request.

## License

This project is licensed under the [MIT License](LICENSE).

