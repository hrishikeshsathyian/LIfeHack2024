import React, { useRef, useEffect, useState } from 'react';
import { MapContainer, TileLayer, useMap, Rectangle } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import 'leaflet-draw/dist/leaflet.draw.css';
import L from 'leaflet';
import 'leaflet-draw';
import '../styles.css'; // Import custom CSS file

const MapComponent = () => {
  const [rectangleCoordinates, setRectangleCoordinates] = useState(null);
  const drawnItemsRef = useRef(new L.FeatureGroup());

  const DrawControl = () => {
    const map = useMap();

    useEffect(() => {
      const drawnItems = drawnItemsRef.current;
      map.addLayer(drawnItems);

      const drawControl = new L.Control.Draw({
        draw: {
          polyline: false,
          polygon: false,
          circle: false,
          marker: false,
          circlemarker: false,
          rectangle: true,
        },
        edit: {
          featureGroup: drawnItems,
        },
      });

      map.addControl(drawControl);

      const onDrawCreated = (e) => {
        const { layer } = e;
        drawnItems.addLayer(layer);
        const { _northEast, _southWest } = layer.getBounds();
        setRectangleCoordinates({
          northEast: _northEast,
          southWest: _southWest,
        });
      };

      map.on(L.Draw.Event.CREATED, onDrawCreated);

      // Cleanup function to remove the draw control and event listener
      return () => {
        map.off(L.Draw.Event.CREATED, onDrawCreated);
        map.removeControl(drawControl);
      };
    }, [map]);

    return null;
  };

  const handleSubmit = () => {
    if (rectangleCoordinates) {
      console.log('Rectangle Coordinates:', rectangleCoordinates);
    } else {
      console.log('No rectangle drawn');
    }
  };

  const GridLayer = () => {
    const map = useMap();
    const [gridCells, setGridCells] = useState([]);
    const [center, setCenter] = useState(map.getCenter()); // Store initial center
  
    useEffect(() => {
      const updateGrid = () => {
        const bounds = L.latLngBounds( // Create bounds around center
          [center.lat - 0.1, center.lng - 0.1],  // Adjust these values for grid size
          [center.lat + 0.1, center.lng + 0.1]
        );
        const cellSize = 500;
  
        const cells = [];
        for (let x = bounds.getWest(); x < bounds.getEast(); x += cellSize / 100000) {
          for (let y = bounds.getSouth(); y < bounds.getNorth(); y += cellSize / 100000) {
            cells.push(
              <Rectangle
                key={`${x}-${y}`}
                bounds={[[y, x], [y + cellSize / 100000, x + cellSize / 100000]]}
                pathOptions={{ weight: 1, color: '#000000', fillOpacity: 0 }}
              />
            );
          }
        }
        setGridCells(cells);
      };
  
      updateGrid();
  
      // Only update on zoom change
      map.on('zoomend', updateGrid); 
      map.on('moveend', updateGrid);
  
      return () => {
        map.off('zoomend', updateGrid);
      };
    }, [map, center]);  // Include center in dependency array
  
    return <>{gridCells}</>;
  };

  return (
    <div>
      <div className="controls">
        <h2>Select a Rectangle on the Map</h2>
        <button onClick={handleSubmit}>Submit</button>
      </div>
      <MapContainer
        center={[1.3521, 103.8198]} // Center on Singapore
        zoom={13}
        id="map"
        style={{ height: '80vh', width: '80%', margin: 'auto' }}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution="&copy; <a href='http://osm.org/copyright'>OpenStreetMap</a> contributors"
        />
        <DrawControl />
        <GridLayer />
      </MapContainer>
    </div>
  );
};

export default MapComponent;
