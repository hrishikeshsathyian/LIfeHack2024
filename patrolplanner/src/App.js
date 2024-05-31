import React, { useState } from "react";
import "./App.css";
import MapComponent from "./components /MapComponent";
import "./styles.css";

function App() {
  const [numOfPatrols, setNumOfPatrols] = useState(0);
  const [file, setFile] = useState(null);
  const [heatmapData, setHeatmapData] = useState(null);
  const [mapCoordinates, setMapCoordinates] = useState(null); // State to hold mapCoordinates

  const handleMapConstantChange = (newmapCoordinates) => {
    setMapCoordinates(newmapCoordinates);
  };

  const handleOnChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleOnSubmit = async (e) => {
    e.preventDefault();

    const formData = new FormData();

    try {
      formData.append("csvFile", file);
      formData.append("topRightLat", mapCoordinates.northEast.lat);
      formData.append("topRightLng", mapCoordinates.northEast.lng);
      formData.append("bottomLeftLat", mapCoordinates.southWest.lat);
      formData.append("bottomLeftLng", mapCoordinates.southWest.lng);
      console.log("FormData:", Array.from(formData.entries()));
      const response = await fetch("http://127.0.0.1:5000/upload", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (response.ok) {
        console.log("Heatmap data obtained");
        setHeatmapData(data); // Update state with heatmap data
      } else {
        console.log("Issue getting response");
        console.error("Error:", data.error); // Handle errors
      }
    } catch (error) {
      console.error("Network Error:", error);
    }
  };

  return (
    <div>
      <div
        style={{
          position: "absolute",
          backgroundColor: "white",
          right: "1%",
          top: "15%",
          padding: 5,
          borderStyle: "solid",
          borderColor: "black",
          borderWidth: 2,
          zIndex: 10000,
        }}
      >
        <p>Upload Crime Data</p>
        <form onSubmit={handleOnSubmit}>
          <input
            type={"file"}
            id={"csvFileInput"}
            accept={".csv"}
            onChange={handleOnChange}
          />
          <button type="submit">Import .csv</button>
        </form>
        <p>Number of patrols:</p>
        <input
          type="number"
          value={numOfPatrols}
          onChange={(e) => setNumOfPatrols(e.target.value)}
        />{" "}
        <div>
          {mapCoordinates && (
            <div>
              Northeast: {mapCoordinates.northEast.lat},{" "}
              {mapCoordinates.northEast.lng}
              <br />
              Southwest: {mapCoordinates.southWest.lat},{" "}
              {mapCoordinates.southWest.lng}
            </div>
          )}
        </div>
      </div>
      <div style={{ zIndex: 0 }}>
        <MapComponent
          onMapConstantChange={handleMapConstantChange}
          heatmapData={heatmapData}
        />
      </div>
    </div>
  );
}

export default App;
