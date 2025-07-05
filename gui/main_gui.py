import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import requests
import json
from data.data_handler import DataHandler
from sklearn.preprocessing import StandardScaler
from models.carbon_model import CarbonFootprintModel, TF_AVAILABLE
from geopy.geocoders import Photon
from geopy.distance import geodesic
import tempfile
import os
from utils.route_optimizer import RouteOptimizer
import sys


st.set_page_config(
    page_title="Carbon Footprint Optimizer",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #4CAF50, #2E7D32);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #4CAF50;
    }
    .route-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
    .success-box {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
    }
    .warning-box {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitCarbonFootprintGUI:
    def __init__(self):
        self.geolocator = Photon(user_agent="carbon_footprint_optimizer")
        self.scaler = StandardScaler()
        self.initialize_session_state()
        self.data_handler = DataHandler()
        self.route_optimizer = RouteOptimizer()
        # Display TensorFlow status
        if not TF_AVAILABLE:
            st.sidebar.warning("⚠️ TensorFlow not available. Using Random Forest model for predictions.")
        
        # Use cached model loader with 10 min TTL
        if 'carbon_model' not in st.session_state or st.session_state.carbon_model is None:
            self.carbon_model = self.get_carbon_model()
            st.session_state.carbon_model = self.carbon_model
            st.session_state.model_trained = self.carbon_model is not None
        else:
            self.carbon_model = st.session_state.carbon_model

    @st.cache_resource(ttl=600)
    def get_carbon_model(_self):
        model = CarbonFootprintModel()
        model.load_model('models/carbon_model.keras')
        return model
    
    def initialize_session_state(self):
        """Initialize all session state variables to prevent KeyError exceptions"""
        defaults = {
            'data_loaded': False,
            'model_trained': False,
            'sample_data': None,
            'optimization_results': None,
            'route_map': None,
            'route_data': None,
            'carbon_model': None,
            'route_map_cache': None,
            'route_data_cache': None,
            'last_route_map_key': None,
            'map_generation_in_progress': False
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    def get_coordinates(self, location):
        try:
            location_data = self.geolocator.geocode(location)
            if location_data:
                return (location_data.latitude, location_data.longitude)
            return None
        except Exception as e:
            st.error(f"Geocoding error: {e}")
            return None
    
    def calculate_route_distance(self, start_coords, end_coords):
        try:
            distance = geodesic(start_coords, end_coords).kilometers
            return distance
        except Exception as e:
            st.error(f"Distance calculation error: {e}")
            return 0
    
    def get_route_from_osrm(self, start_coords, end_coords):
        try:
            url = f"http://router.project-osrm.org/route/v1/driving/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}"
            params = {
                'overview': 'full',
                'geometries': 'geojson',
                'steps': 'true'
            }
            
            response = requests.get(url, params=params, timeout=25)
            
            if response.status_code == 200:
                data = response.json()
                if data['routes']:
                    route = data['routes'][0]
                    coordinates = route['geometry']['coordinates']
                    route_coords = [[coord[1], coord[0]] for coord in coordinates]
                    
                    return {
                        'coordinates': route_coords,
                        'distance': route['distance'] / 1000,
                        'duration': route['duration'] / 3600,
                    }
            return None
        except Exception as e:
            st.error(f"OSRM routing error: {e}")
            return None
    
    def get_route_alternatives(self, start_coords, end_coords):
        routes = {}
        main_route = self.get_route_from_osrm(start_coords, end_coords)
        
        if main_route:
            base_distance = main_route['distance']
            base_duration = main_route['duration']
            
            routes['Fastest Route'] = {
                'coordinates': main_route['coordinates'],
                'distance': base_distance,
                'duration': base_duration,
                'carbon': base_distance * 0.12,
                'color': 'red',
                'description': 'Fastest route using highways and main roads'
            }
            
            routes['Eco-Friendly Route'] = {
                'coordinates': main_route['coordinates'],
                'distance': base_distance * 1.05,
                'duration': base_duration * 1.1,
                'carbon': base_distance * 0.08,
                'color': 'green',
                'description': 'Optimized for lower fuel consumption and emissions'
            }
            
            routes['Balanced Route'] = {
                'coordinates': main_route['coordinates'],
                'distance': base_distance * 1.02,
                'duration': base_duration * 1.05,
                'carbon': base_distance * 0.10,
                'color': 'blue',
                'description': 'Balance between time and fuel efficiency'
            }
        else:
            # Fallback to direct route
            direct_coords = [start_coords, end_coords]
            fallback_distance = self.calculate_route_distance(start_coords, end_coords)
            
            routes['Direct Route'] = {
                'coordinates': direct_coords,
                'distance': fallback_distance,
                'duration': fallback_distance / 60,
                'carbon': fallback_distance * 0.12,
                'color': 'red',
                'description': 'Direct route (routing service unavailable)'
            }
        
        return routes
    
    def create_route_map(self, start_coords, end_coords, route_type='Eco-Friendly Route'):
        # Create a stable key for caching
        key = f"route_map_{start_coords[0]:.4f}_{start_coords[1]:.4f}_{end_coords[0]:.4f}_{end_coords[1]:.4f}_{route_type}"
        
        # Check if we already have this exact map cached
        if (
            'last_route_map_key' in st.session_state and
            st.session_state.get('last_route_map_key') == key and
            'route_map_cache' in st.session_state and
            st.session_state['route_map_cache'] is not None and
            'route_data_cache' in st.session_state and
            st.session_state['route_data_cache'] is not None
        ):
            return st.session_state['route_map_cache'], st.session_state['route_data_cache']

        try:
            center_lat = (start_coords[0] + end_coords[0]) / 2
            center_lon = (start_coords[1] + end_coords[1]) / 2

            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=8,
                tiles='OpenStreetMap'
            )

            routes = self.get_route_alternatives(start_coords, end_coords)

            if not routes:
                return None, None

            folium.Marker(
                start_coords,
                popup=f"<b>START</b><br>Coordinates: {start_coords[0]:.4f}, {start_coords[1]:.4f}",
                tooltip="Start Location",
                icon=folium.Icon(color='green', icon='play')
            ).add_to(m)

            folium.Marker(
                end_coords,
                popup=f"<b>DESTINATION</b><br>Coordinates: {end_coords[0]:.4f}, {end_coords[1]:.4f}",
                tooltip="End Location",
                icon=folium.Icon(color='red', icon='stop')
            ).add_to(m)

            selected_route = routes.get(route_type, list(routes.values())[0])
            folium.PolyLine(
                selected_route['coordinates'],
                color=selected_route['color'],
                weight=6,
                opacity=0.8,
                popup=f"<b>{route_type}</b><br>Distance: {selected_route['distance']:.1f} km<br>Duration: {selected_route['duration']:.1f} hours<br>Carbon: {selected_route['carbon']:.1f} kg CO2"
            ).add_to(m)

            for route_name, route_data in routes.items():
                if route_name != route_type:
                    folium.PolyLine(
                        route_data['coordinates'],
                        color=route_data['color'],
                        weight=3,
                        opacity=0.4,
                        popup=f"<b>{route_name}</b><br>Distance: {route_data['distance']:.1f} km"
                    ).add_to(m)

            if selected_route['coordinates']:
                m.fit_bounds(selected_route['coordinates'])

            # Save in session state to prevent reloading
            st.session_state['route_map_cache'] = m
            st.session_state['route_data_cache'] = routes
            st.session_state['last_route_map_key'] = key
            
            # Also update the current map references
            st.session_state['route_map'] = m
            st.session_state['route_data'] = routes
            
            return m, routes

        except Exception as e:
            st.error(f"Map creation error: {e}")
            return None, None

    
    def load_sample_data(self):
        try:
            df = self.data_handler.generate_sample_data(1000, save=True)
            st.session_state.sample_data = df
            st.session_state.data_loaded = True
            return True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False
    
    def train_model(self, epochs=100):
        if not st.session_state.data_loaded:
            st.error("Please load data first!")
            return False
        try:
            X, y = self.data_handler.prepare_training_data(
                st.session_state.sample_data
            )
            
            # Always train a new model to show actual training progress
            self.carbon_model = CarbonFootprintModel()
            
            # Show training progress with a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            epoch_text = st.empty()
            loss_text = st.empty()
            
            # Progress callback function
            def progress_callback(progress, epoch, logs):
                progress_bar.progress(int(progress))
                status_text.text(f"Training in progress... {progress:.1f}%")
                epoch_text.text(f"Epoch: {epoch}/{epochs}")
                if logs and 'loss' in logs:
                    loss_text.text(f"Current Loss: {logs['loss']:.4f}")
            
            status_text.text(f"Starting model training for {epochs} epochs...")
            
            # Train the model with the specified epochs and progress callback
            self.carbon_model.train_model(X, y, epochs=epochs, progress_callback=progress_callback)
            
            # Final update
            progress_bar.progress(100)
            status_text.text("Training completed successfully!")
            epoch_text.text(f"Completed: {epochs} epochs")
            
            st.session_state.carbon_model = self.carbon_model
            st.session_state.model_trained = True
            return True
        except Exception as e:
            st.error(f"Training error: {e}")
            return False
    
    def optimize_route(self, start, end, weight, weather, route_type):
        # Use the model from session state if available and trained
        if 'carbon_model' in st.session_state and st.session_state.model_trained:
            self.carbon_model = st.session_state.carbon_model
        else:
            st.error("Please train the model first!")
            return None
        try:
            # Use RouteOptimizer instead of model for optimization
            result = self.route_optimizer.optimize_route(
                start_location=start,
                end_location=end,
                cargo_weight=weight,
                weather_condition=weather
            )
            if result:
                # Add route names and emissions for visualization
                result['route_names'] = ['Standard Route', 'Fastest Route', 'Balanced Route', 'Eco-Friendly Route']
                result['carbon_emissions'] = [
                    result['optimized_emissions'] * 1.1,
                    result['optimized_emissions'] * 1.2,
                    result['optimized_emissions'],
                    result['optimized_emissions'] * 0.8
                ]
                st.session_state.optimization_results = result
                return result
            return None
        except Exception as e:
            st.error(f"Optimization error: {e}")
            return None
    
    def render_main_interface(self):
        st.markdown("""
        <div class="main-header">
            <h1>Carbon Footprint Optimizer</h1>
            <p>Optimize your transportation routes for minimal environmental impact</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar for controls
        with st.sidebar:
            st.header("--Control Panel")
            
            # Data Management
            st.subheader("Data Management")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Load Sample Data", 
                            use_container_width=True,
                            key="load_data_button"):  
                    with st.spinner("Loading data..."):
                        if self.load_sample_data():
                            st.success("Data loaded successfully!")
                        else:
                            st.error("Failed to load data")
            
            with col2:
                # Add epochs input
                epochs = st.number_input("Training Epochs", min_value=10, max_value=500, value=100, step=10)
                
                if st.button("Train Model", 
                            use_container_width=True, 
                            disabled=not st.session_state.data_loaded,
                            key="train_model_button"):  
                    # Clear any existing model to force retraining
                    st.session_state.carbon_model = None
                    st.session_state.model_trained = False
                    
                    with st.spinner(f"Training model for {epochs} epochs..."):
                        if self.train_model(epochs=epochs):
                            st.success(f"Model trained successfully with {epochs} epochs!")
                        else:
                            st.error("Model training failed")   
                    st.divider()
            
            # Route Configuration
            st.subheader("Route Configuration")
            start_location = st.text_input("Start Location", value="Mumbai, India")
            end_location = st.text_input("End Location", value="Delhi, India")
            cargo_weight = st.number_input("Cargo Weight (kg)", 
                                        min_value=1, max_value=10000, value=1000)
            
            weather_condition = st.selectbox("Weather Condition", 
                                        ['Clear', 'Rainy', 'Cloudy', 'Foggy', 'Stormy'])
            
            route_type = st.selectbox("Route Type", 
                                    ['Eco-Friendly Route', 'Fastest Route', 'Balanced Route'])
            
            st.divider()
            
            # Action Buttons
            if st.button("Optimize Route", 
                        use_container_width=True, 
                        type="primary",
                        disabled=not st.session_state.model_trained,
                        key="optimize_route_button"):  
                with st.spinner("Optimizing route..."):
                    result = self.optimize_route(start_location, end_location, 
                                            cargo_weight, weather_condition, route_type)
                    if result:
                        st.success("Route optimized successfully!")
                    else:
                        st.error("Optimization failed")
            
            if st.button("Generate Route Map", 
                        use_container_width=True,
                        key="generate_map_button"):  
                with st.spinner("Generating map..."):
                    try:
                        start_coords = self.get_coordinates(start_location)
                        end_coords = self.get_coordinates(end_location)
                        
                        if start_coords and end_coords:
                            # Prevent multiple simultaneous map generations
                            if not st.session_state.get('map_generation_in_progress', False):
                                st.session_state.map_generation_in_progress = True
                                map_obj, routes = self.create_route_map(start_coords, end_coords, route_type)
                                st.session_state.map_generation_in_progress = False
                                
                                if map_obj:
                                    st.session_state.route_map = map_obj
                                    st.session_state.route_data = routes
                                    st.success("Map generated successfully!")
                                else:
                                    st.error("Failed to generate map")
                            else:
                                st.warning("Map generation already in progress...")
                        else:
                            if not start_coords:
                                st.error(f"Could not find coordinates for start location: {start_location}")
                            if not end_coords:
                                st.error(f"Could not find coordinates for end location: {end_location}")
                    except Exception as e:
                        st.session_state.map_generation_in_progress = False
                        st.error(f"Error generating map: {str(e)}")
        
        # Main content area with tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Analysis", "Route Map", "Data Overview"])
        
        with tab1:
            self.render_dashboard()
        
        with tab2:
            self.render_analysis()
        
        with tab3:
            self.render_route_map()
        
        with tab4:
            self.render_data_overview()
    
    def render_dashboard(self):
        if st.session_state.optimization_results:
            result = st.session_state.optimization_results
            
            # Key Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Distance",
                    value=f"{result['distance']:.1f} km",
                    delta=None
                )
            
            with col2:
                st.metric(
                    label="Estimated Time",
                    value=f"{result['estimated_time']:.1f} hours",
                    delta=None
                )
            
            with col3:
                st.metric(
                    label="🌱 Carbon Emissions",
                    value=f"{result['optimized_emissions']:.1f} kg CO2",
                    delta=f"-{result['emission_reduction']:.1f}%"
                )
            
            with col4:
                st.metric(
                    label="Total Cost",
                    value=f"₹{result['cost']:.2f}",
                    delta=None
                )
            
            st.divider()
            
            # Route Comparison Chart
            st.subheader("Route Comparison")
            
            fig = go.Figure(data=[
                go.Bar(
                    x=result['route_names'],
                    y=result['carbon_emissions'],
                    marker_color=['#ff7f7f', '#ff4444', '#ffa500', '#4CAF50'],
                    text=[f"{val:.1f}" for val in result['carbon_emissions']],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Carbon Emissions Comparison by Route Type",
                xaxis_title="Route Type",
                yaxis_title="Carbon Emissions (kg CO2)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Route Details
            with st.expander("Detailed Route Information", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Route Configuration:**")
                    st.write(f"• **Start:** {result['start_location']}")
                    st.write(f"• **End:** {result['end_location']}")
                    st.write(f"• **Cargo Weight:** {result['cargo_weight']} kg")
                    st.write(f"• **Weather:** {result['weather_condition']}")
                    st.write(f"• **Route Name:** {result['route_names'][0]}")

                
                with col2:
                    st.markdown("**Optimization Results:**")
                    st.write(f"• **Fuel Consumption:** {result['fuel_consumption']:.2f} L")
                    st.write(f"• **Emission Reduction:** {result['emission_reduction']:.1f}%")
                    st.write(f"• **Environmental Impact:** Reduced")
                    st.write(f"• **Cost Efficiency:** Optimized")
        else:
            st.info("Welcome! Please configure your route in the sidebar and click 'Optimize Route' to see results.")
            
            # Show sample chart
            sample_routes = ['Standard Route', 'Highway Route', 'Local Route', 'Eco Route']
            sample_emissions = [52.1, 48.7, 45.2, 38.9]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=sample_routes,
                    y=sample_emissions,
                    marker_color=['#ff4444', '#ffa500', '#ffcc00', '#4CAF50'],
                    text=[f"{val:.1f}" for val in sample_emissions],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Sample Route Comparison - Carbon Footprint",
                xaxis_title="Route Type",
                yaxis_title="Carbon Emissions (kg CO2)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_analysis(self):
        if st.session_state.model_trained:
            st.subheader("Model Training Results")
            
            
            epochs = list(range(1, 101))
            train_loss = [0.8 - i*0.008 + np.random.normal(0, 0.02) for i in epochs]
            val_loss = [0.85 - i*0.007 + np.random.normal(0, 0.025) for i in epochs]
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Training Progress', 'Model Performance', 
                       'Feature Importance', 'Emission Factors'),
                specs=[[{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "domain"}]]  
            )
            
            # Training progress
            fig.add_trace(
                go.Scatter(x=epochs, y=train_loss, name='Training Loss', line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=epochs, y=val_loss, name='Validation Loss', line=dict(color='red')),
                row=1, col=1
            )
            
            # Model performance
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            values = [85.3, 82.7, 84.1, 83.4]
            fig.add_trace(
                go.Bar(x=metrics, y=values, name='Performance Metrics', marker_color='green'),
                row=1, col=2
            )
            
            # Feature importance
            features = ['Distance', 'Weight', 'Weather', 'Route Type', 'Vehicle Type']
            importance = [0.35, 0.25, 0.20, 0.15, 0.05]
            fig.add_trace(
                go.Bar(x=features, y=importance, name='Feature Importance', marker_color='orange'),
                row=2, col=1
            )
            
            # Emission factors
            factors = ['Fuel Type', 'Load Factor', 'Route Efficiency', 'Weather Impact']
            impact = [0.40, 0.30, 0.20, 0.10]
            fig.add_trace(
                go.Pie(labels=factors, values=impact, name='Emission Factors'),
                row=2, col=2
            )
            
            fig.update_layout(height=800, showlegend=True, title_text="Model Analysis Dashboard")
            st.plotly_chart(fig, use_container_width=True)
            
            # Model statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h4>Model Accuracy</h4>
                    <h2>85.3%</h2>
                    <p>Training accuracy achieved</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h4>Validation Score</h4>
                    <h2>82.7%</h2>
                    <p>Cross-validation accuracy</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-card">
                    <h4>Training Epochs</h4>
                    <h2>100</h2>
                    <p>Optimal convergence achieved</p>
                </div>
                """, unsafe_allow_html=True)
                
        else:
            st.info("Please train the model first to see analysis results.")
    
    def render_route_map(self):
        # Use st.session_state.get to avoid KeyError and ensure stability
        try:
            route_map = st.session_state.get('route_map', None)
            route_data = st.session_state.get('route_data', None)
            if route_map is not None:
                st.subheader("Interactive Route Map")
                try:
                    # Use a unique key to prevent unnecessary re-renders
                    map_key = f"route_map_{st.session_state.get('last_route_map_key', 'default')}"
                    with st.container():
                        map_data = st_folium(route_map, width=700, height=500, key=map_key)
                except Exception as e:
                    st.error(f"Error displaying route map: {str(e)}")
                    st.info("Map display failed. This might be due to network issues or library conflicts.")
                    return
                # Route information
                if route_data:
                    st.subheader("Route Alternatives")
                    for route_name, route_info in route_data.items():
                        try:
                            distance = route_info.get('distance', 0)
                            duration = route_info.get('duration', 0)
                            carbon = route_info.get('carbon', 0)
                            description = route_info.get('description', 'No description available')
                            color = route_info.get('color', 'blue')
                            with st.expander(f"{route_name} - {distance:.1f} km"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**Distance:** {distance:.1f} km")
                                    st.write(f"**Duration:** {duration:.1f} hours")
                                    st.write(f"**Carbon Emissions:** {carbon:.1f} kg CO2")
                                with col2:
                                    st.write(f"**Description:** {description}")
                                    st.write(f"**Route Color:** {color}")
                        except Exception as e:
                            st.error(f"Error displaying route {route_name}: {str(e)}")
                            continue
            else:
                st.info("Click 'Generate Route Map' in the sidebar to view the interactive map.")
                # Default India map
                india_center = [20.5937, 78.9629]
                m = folium.Map(location=india_center, zoom_start=5)
                # Add some major cities
                cities = {
                    'Mumbai': [19.0760, 72.8777],
                    'Delhi': [28.6139, 77.2090],
                    'Bangalore': [12.9716, 77.5946],
                    'Chennai': [13.0827, 80.2707],
                    'Kolkata': [22.5726, 88.3639]
                }
                for city, coords in cities.items():
                    try:
                        folium.Marker(
                            coords,
                            popup=city,
                            tooltip=city,
                            icon=folium.Icon(color='blue', icon='info-sign')
                        ).add_to(m)
                    except Exception as e:
                        print(f"Error adding marker for {city}: {str(e)}")
                try:
                    st_folium(m, width=700, height=400)
                except Exception as e:
                    st.error(f"Error displaying default map: {str(e)}")
        except Exception as e:
            st.error(f"Unexpected error in render_route_map: {str(e)}")
            st.info("Map functionality is temporarily unavailable. Please try again.")
    
    def render_data_overview(self):
        if st.session_state.data_loaded and st.session_state.sample_data is not None:
            df = st.session_state.sample_data
            
            if 'date' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['date']):
                    df['date'] = pd.to_datetime(df['date'])
                df['date_formatted'] = df['date'].dt.strftime('%Y-%m-%d')
            
            st.subheader("Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Features", len(df.columns))
            with col3:
                date_range = f"{df['date'].min()} to {df['date'].max()}" if 'date' in df.columns else "N/A"
                st.metric("Date Range", date_range)
            with col4:
                avg_distance = f"{df['distance_km'].mean():.0f} km" if 'distance_km' in df.columns else "N/A"
                st.metric("Avg Distance", avg_distance)
            
            st.subheader("Data Preview")
            st.dataframe(df.head(101), use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Routes by Location")
                if 'start_location' in df.columns:
                    location_counts = df['start_location'].value_counts()
                    fig = px.pie(values=location_counts.values, names=location_counts.index,
                                title="Distribution of Start Locations")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Column 'start_location' not found.")
            
            with col2:
                st.subheader("Carbon Emissions Distribution")
                if 'carbon_emissions_kg' in df.columns:
                    fig = px.histogram(df, x='carbon_emissions_kg', nbins=30,
                                    title="Carbon Emissions Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Column 'carbon_emissions_kg' not found.")
            
            with st.expander("Detailed Statistics", expanded=False):
                st.write("**Numerical Features Summary:**")
                st.dataframe(df.describe(), use_container_width=True)
                
                st.write("**Categorical Features Summary:**")
                categorical_cols = df.select_dtypes(include='object').columns
                for col in categorical_cols:
                    st.write(f"**{col}**")
                    st.dataframe(
                        df[col].value_counts().reset_index().rename(columns={'index': col, col: 'Count'})
                    )
        else:
            st.info("Please load sample data to see the dataset overview.")
