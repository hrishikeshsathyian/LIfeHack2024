# Filename - app.py
 
# Import flask and datetime module for showing date and time
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import datetime
import heatmap
 
x = datetime.datetime.now()

# Initializing flask app
app = Flask(__name__)

CORS(app, resources={
    r"/upload": {
        "origins": ["http://localhost:3000"],  # Whitelist the frontend origin
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_csv():
    if request.method == 'OPTIONS':
        response = jsonify()
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response = Response(heatmap_data)
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    
    try:
        file = request.files['csvFile']  # Access the file sent from frontend
        topRightLat = float(request.form.get('topRightLat'))
        topRightLng = float(request.form.get('topRightLng'))
        bottomLeftLat = float(request.form.get('bottomLeftLat'))
        bottomLeftLng = float(request.form.get('bottomLeftLng'))
        topRight = [topRightLat, topRightLng]
        bottomLeft = [bottomLeftLat, bottomLeftLng]

        # Call your heatmap function
        grid_coords, z_values = heatmap.generate_heatmap(file, topRight, bottomLeft)

        # Prepare data for sending back to frontend
        heatmap_data = {
            'gridCoords': grid_coords.tolist(),
            'zValues': z_values.tolist()
        }
        return jsonify(heatmap_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Handle errors gracefully
 
# Route for seeing a data
@app.route('/data')
def get_time():
 
    # Returning an api for showing in  reactjs
    return {
        'Name':"geek", 
        "Age":"22",
        "Date":x, 
        "programming":"python"
        }
 
# Running app
if __name__ == '__main__':
    app.run(debug=True)