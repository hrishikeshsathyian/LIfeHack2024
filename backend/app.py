import logging
from flask import Flask, request, jsonify
from route_optimiser import *
import os
# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Ensure the uploaded_files directory exists
UPLOAD_DIR = 'uploaded_files'
os.makedirs(UPLOAD_DIR, exist_ok=True)
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'csvFile' not in request.files:
        logging.error("No file part in the request")
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['csvFile']
    if file.filename == '':
        logging.error("No selected file")
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        try:
            # Save file
            file_path = os.path.join("uploaded_files", file.filename)
            file.save(file_path)

            # Get additional form data
            top_right_lat = float(request.form['topRightLat'])
            top_right_lng = float(request.form['topRightLng'])
            bottom_left_lat = float(request.form['bottomLeftLat'])
            bottom_left_lng = float(request.form['bottomLeftLng'])
            num_patrols = int(request.form['numPatrols'])
            
            upper_right = (top_right_lat, top_right_lng)
            bottom_left = (bottom_left_lat, bottom_left_lng)
            severity_weight = 1.5
            time_weight = 0.1
            n_clusters = num_patrols
            output_dir = 'plot_images'
            
            os.makedirs(output_dir, exist_ok=True)

            rectangles, grid_coords, heatmap_values, cluster_labels, cluster_centers = get_cluster_rectangles(file_path, upper_right, bottom_left, severity_weight, time_weight, n_clusters, output_dir)
            plot_osmnx_map_with_rectangles(grid_coords, rectangles, output_dir, "osmnx_map")

            for i, rectangle in enumerate(rectangles):
                plot_osmnx_map_with_intensity_route(grid_coords, heatmap_values, rectangle, output_dir, f"osmnx_intensity_route_{i}")

            logging.info("Files processed and plots created successfully")
            return jsonify({"message": "Files processed and plots created successfully"}), 200

        except Exception as e:
            logging.error("Error processing file: %s", str(e))
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
