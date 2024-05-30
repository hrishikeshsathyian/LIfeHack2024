import React, { useRef, useEffect, useState } from 'react';
import { MapContainer, TileLayer, useMap } from 'react-leaflet';
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
      </MapContainer>
    </div>
  );
};

export default MapComponent;
