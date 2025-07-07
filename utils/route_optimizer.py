import numpy as np
import pandas as pd
import random
from datetime import datetime
import math
from geopy.geocoders import Photon


class RouteOptimizer:
    
    def __init__(self, api_key=None):
        self.api_key = api_key or "DEMO_API_KEY"
        self.base_url = "https://maps.googleapis.com/maps/api"
        
        self.geolocator = Photon(user_agent="carbon_footprint_optimizer")
        
        self.vehicle_efficiency = {
            'Truck': 4.5,
            'Van': 8.0,
            'Container': 3.2
        }
        
        self.emission_factors = {
            'Diesel': 2.68,
            'Petrol': 2.31,
            'CNG': 1.94,
            'Electric': 0.5
        }
        
        self.weather_factors = {
            'Clear': 1.0,
            'Cloudy': 1.05,
            'Rainy': 1.2,
            'Foggy': 1.15,
            'Stormy': 1.3
        }
        
        self.traffic_factors = {
            'Low': 1.0,
            'Medium': 1.15,
            'High': 1.3
        }
    
        self.city_coordinates = {
            'Mumbai, India': {'lat': 19.0760, 'lng': 72.8777},
            'Delhi, India': {'lat': 28.7041, 'lng': 77.1025},
            'Bangalore, India': {'lat': 12.9716, 'lng': 77.5946},
            'Chennai, India': {'lat': 13.0827, 'lng': 80.2707},
            'Kolkata, India': {'lat': 22.5726, 'lng': 88.3639},
            'Hyderabad, India': {'lat': 17.3850, 'lng': 78.4867},
            'Pune, India': {'lat': 18.5204, 'lng': 73.8567},
            'Ahmedabad, India': {'lat': 23.0225, 'lng': 72.5714},
            'Jaipur, India': {'lat': 26.9124, 'lng': 75.7873},
            'Lucknow, India': {'lat': 26.8467, 'lng': 80.9462},
            'Mumbai': {'lat': 19.0760, 'lng': 72.8777},
            'Delhi': {'lat': 28.7041, 'lng': 77.1025},
            'Bangalore': {'lat': 12.9716, 'lng': 77.5946},
            'Chennai': {'lat': 13.0827, 'lng': 80.2707},
            'Kolkata': {'lat': 22.5726, 'lng': 88.3639},
            'Hyderabad': {'lat': 17.3850, 'lng': 78.4867},
            'Pune': {'lat': 18.5204, 'lng': 73.8567},
            'Ahmedabad': {'lat': 23.0225, 'lng': 72.5714},
            'Jaipur': {'lat': 26.9124, 'lng': 75.7873},
            'Lucknow': {'lat': 26.8467, 'lng': 80.9462}
        }
    
    def get_coordinates(self, location):
        try:
            if location in self.city_coordinates:
                coords = self.city_coordinates[location]
                return (coords['lat'], coords['lng'])
            
            location_data = self.geolocator.geocode(location)
            if location_data:
                coords = (location_data.latitude, location_data.longitude)
                self.city_coordinates[location] = {'lat': coords[0], 'lng': coords[1]}
                return coords
            
            return None
        except Exception as e:
            print(f"Geocoding error for {location}: {e}")
            return None
    
    def get_route_from_osrm(self, start_coords, end_coords):
        try:
            import requests
            url = f"http://router.project-osrm.org/route/v1/driving/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}"
            params = {
                'overview': 'full',
                'geometries': 'geojson',
                'steps': 'true'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data['routes']:
                    route = data['routes'][0]
                    coordinates = route['geometry']['coordinates']
                    route_coords = [[coord[1], coord[0]] for coord in coordinates]
                    
                    return {
                        'coordinates': route_coords,
                        'distance_km': route['distance'] / 1000,  # Convert to km
                        'duration_hours': route['duration'] / 3600,  # Convert to hours
                        'status': 'OSRM_OK'
                    }
            return None
        except Exception as e:
            print(f"OSRM routing error: {e}")
            return None
    
    def get_distance_and_duration(self, start_location, end_location):
        try:
            start_coords = self.get_coordinates(start_location)
            end_coords = self.get_coordinates(end_location)
            if not start_coords or not end_coords:
                print(f"Could not find coordinates for {start_location} or {end_location}")
                return None
            
            route_data = self.get_route_from_osrm(start_coords, end_coords)
            if route_data:
                return route_data
            
            distance = self.calculate_haversine_distance(
                start_coords[0], start_coords[1],
                end_coords[0], end_coords[1]
            )
            
            duration = distance / 60.0
            
            return {
                'distance_km': distance,
                'duration_hours': duration,
                'status': 'HAVERSINE_FALLBACK',
                'coordinates': [list(start_coords), list(end_coords)]
            }

        except Exception as e:
            print(f"Error getting distance: {e}")
            return None
    
    def calculate_haversine_distance(self, lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        r = 6371
        
        return c * r
    
    def get_alternative_routes(self, start_location, end_location):
        routes = []
        base_route = self.get_distance_and_duration(start_location, end_location)
        
        if base_route:
            route_types = [
                ('Fastest Route', 1.0, 0.9, 'High'),
                ('Eco-Friendly Route', 1.1, 1.2, 'Low'),
                ('Balanced Route', 1.05, 1.05, 'Medium')
            ]
            
            for route_name, distance_factor, duration_factor, traffic_level in route_types:
                route = {
                    'name': route_name,
                    'distance_km': base_route['distance_km'] * distance_factor,
                    'duration_hours': base_route['duration_hours'] * duration_factor,
                    'traffic_level': traffic_level,
                    'road_type': random.choice(['Highway', 'City Roads', 'Mixed']),
                    'tolls': random.choice([True, False]),
                    'coordinates': base_route.get('coordinates', [])
                }
                routes.append(route)
        
        return routes
    
    def calculate_carbon_emissions(self, distance, cargo_weight, vehicle_type='Truck', 
                                 fuel_type='Diesel', weather_condition='Clear', 
                                 traffic_density='Medium'):
        efficiency = self.vehicle_efficiency.get(vehicle_type, 4.5)
        base_fuel_consumption = distance / efficiency
        
        weather_factor = self.weather_factors.get(weather_condition, 1.0)
        traffic_factor = self.traffic_factors.get(traffic_density, 1.15)
        weight_factor = 1 + (cargo_weight / 10000)
        
        fuel_consumption = base_fuel_consumption * weather_factor * traffic_factor * weight_factor
        
        emission_factor = self.emission_factors.get(fuel_type, 2.68)
        carbon_emissions = fuel_consumption * emission_factor
        
        return {
            'fuel_consumption_liters': fuel_consumption,
            'carbon_emissions_kg': carbon_emissions,
            'weather_factor': weather_factor,
            'traffic_factor': traffic_factor,
            'weight_factor': weight_factor
        }
    
    def optimize_route(self, start_location, end_location, cargo_weight, 
                      weather_condition, trained_model=None):
        try:
            routes = self.get_alternative_routes(start_location, end_location)
            
            if not routes:
                raise Exception("Could not generate routes")
            
            route_analysis = []
            for route in routes:
                vehicle_fuel_combinations = [
                    ('Truck', 'Diesel'),
                    ('Van', 'CNG'),
                    ('Container', 'Diesel'),
                    ('Truck', 'Electric')
                ]
                
                for vehicle_type, fuel_type in vehicle_fuel_combinations:
                    emissions_data = self.calculate_carbon_emissions(
                        route['distance_km'], cargo_weight, vehicle_type,
                        fuel_type, weather_condition, route['traffic_level']
                    )
                    
                    fuel_cost_per_liter = {'Diesel': 90, 'Petrol': 100, 'CNG': 60, 'Electric': 30}
                    cost = (route['distance_km'] * 12 + 
                           emissions_data['fuel_consumption_liters'] * fuel_cost_per_liter.get(fuel_type, 90) +
                           cargo_weight * 0.5)
                    
                    analysis = {
                        'route_name': f"{route['name']} ({vehicle_type}-{fuel_type})",
                        'distance_km': route['distance_km'],
                        'duration_hours': route['duration_hours'],
                        'vehicle_type': vehicle_type,
                        'fuel_type': fuel_type,
                        'carbon_emissions_kg': emissions_data['carbon_emissions_kg'],
                        'fuel_consumption_liters': emissions_data['fuel_consumption_liters'],
                        'cost_inr': cost,
                        'traffic_level': route['traffic_level']
                    }
                    route_analysis.append(analysis)
            
            if not route_analysis:
                raise Exception("No route analysis data generated")
            
            route_analysis.sort(key=lambda x: x['carbon_emissions_kg'])
            
            best_route = route_analysis[0]
            standard_route = route_analysis[len(route_analysis)//2]
            
            if standard_route['carbon_emissions_kg'] > 0:
                emission_reduction = ((standard_route['carbon_emissions_kg'] - best_route['carbon_emissions_kg']) / 
                                    standard_route['carbon_emissions_kg']) * 100
            else:
                emission_reduction = 0.0
            cost_savings = standard_route['cost_inr'] - best_route['cost_inr']
            fuel_savings = standard_route['fuel_consumption_liters'] - best_route['fuel_consumption_liters']
            
            result = {
                'start_location': start_location,
                'end_location': end_location,
                'cargo_weight': cargo_weight,
                'weather_condition': weather_condition,
                'distance': best_route['distance_km'],
                'estimated_time': best_route['duration_hours'],
                'fuel_consumption': best_route['fuel_consumption_liters'],
                'optimized_emissions': best_route['carbon_emissions_kg'],
                'cost': best_route['cost_inr'],
                'vehicle_type': best_route['vehicle_type'],
                'fuel_type': best_route['fuel_type'],
                'emission_reduction': max(0,emission_reduction),
                'cost_savings': cost_savings,
                'fuel_savings': fuel_savings,
                'route_names': [r['route_name'] for r in route_analysis[:4]],
                'carbon_emissions': [r['carbon_emissions_kg'] for r in route_analysis[:4]],
                'route_type': best_route['route_name']
            }
            
            return result
            
        except Exception as e:
            print(f"Route optimization error: {e}")
            return None
    
    def get_real_time_traffic(self, start_location, end_location):
        traffic_levels = ['Low', 'Medium', 'High']
        current_traffic = random.choice(traffic_levels)
        
        base_time = self.get_distance_and_duration(start_location, end_location)
        if base_time:
            traffic_multiplier = self.traffic_factors[current_traffic]
            actual_time = base_time['duration_hours'] * traffic_multiplier
            delay = actual_time - base_time['duration_hours']
            
            return {
                'traffic_level': current_traffic,
                'base_duration_hours': base_time['duration_hours'],
                'actual_duration_hours': actual_time,
                'delay_hours': delay,
                'delay_percentage': (delay / base_time['duration_hours']) * 100
            }
        
        return None
    
    def get_weather_impact(self,weather_condition):
        base_factor = self.weather_factors.get(weather_condition, 1.0)
        
        actual_factor = base_factor * random.uniform(0.95, 1.05)
        
        impact_description = {
            'Clear': 'Optimal driving conditions',
            'Cloudy': 'Slightly reduced visibility',
            'Rainy': 'Increased fuel consumption due to wet roads',
            'Foggy': 'Reduced speed due to low visibility',
            'Stormy': 'Severe weather conditions, high fuel consumption'
        }
        
        return {
            'weather_condition': weather_condition,
            'fuel_impact_factor': actual_factor,
            'additional_consumption_percent': (actual_factor - 1) * 100,
            'description': impact_description.get(weather_condition, 'Unknown weather impact')
        }
    
    def calculate_route_efficiency_score(self, route_data):
        baseline_emissions = 50.0
        baseline_distance = 500.0
        baseline_cost = 5000.0
        
        emission_score = max(0, 100 - (route_data['carbon_emissions_kg'] / baseline_emissions) * 100)
        distance_score = max(0, 100 - (route_data['distance_km'] / baseline_distance) * 100)
        cost_score = max(0, 100 - (route_data['cost_inr'] / baseline_cost) * 100)
        
        efficiency_score = (emission_score * 0.5 + distance_score * 0.3 + cost_score * 0.2)
        
        return min(100, max(0, efficiency_score))
    
    def generate_route_report(self, optimization_result):
        if not optimization_result:
            return "No optimization result available"
        
        report = f"""
CARBON FOOTPRINT ROUTE OPTIMIZATION REPORT
==========================================
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

ROUTE DETAILS:
--------------
Origin: {optimization_result['start_location']}
Destination: {optimization_result['end_location']}
Cargo Weight: {optimization_result['cargo_weight']:.1f} kg
Weather Condition: {optimization_result['weather_condition']}

OPTIMIZED ROUTE PERFORMANCE:
---------------------------
Distance: {optimization_result['distance']:.1f} km
Estimated Travel Time: {optimization_result['estimated_time']:.1f} hours
Vehicle Type: {optimization_result['vehicle_type']}
Fuel Type: {optimization_result['fuel_type']}

ENVIRONMENTAL IMPACT:
--------------------
Carbon Emissions: {optimization_result['optimized_emissions']:.2f} kg CO2
Fuel Consumption: {optimization_result['fuel_consumption']:.2f} liters
Emission Reduction: {optimization_result['emission_reduction']:.1f}%

ECONOMIC ANALYSIS:
-----------------
Total Cost: ₹{optimization_result['cost']:.2f}
Cost Savings: ₹{optimization_result['cost_savings']:.2f}
Fuel Savings: {optimization_result['fuel_savings']:.2f} liters

SUSTAINABILITY RATING: {self.calculate_route_efficiency_score(optimization_result):.1f}/100
"""
        
        return report
    
    def save_optimization_history(self, optimization_result, filename='data/optimization_history.csv'):
        try:
            data = {
                'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                'start_location': [optimization_result['start_location']],
                'end_location': [optimization_result['end_location']],
                'cargo_weight': [optimization_result['cargo_weight']],
                'weather_condition': [optimization_result['weather_condition']],
                'distance_km': [optimization_result['distance']],
                'carbon_emissions_kg': [optimization_result['optimized_emissions']],
                'fuel_consumption_liters': [optimization_result['fuel_consumption']],
                'cost_inr': [optimization_result['cost']],
                'emission_reduction_percent': [optimization_result['emission_reduction']],
                'vehicle_type': [optimization_result['vehicle_type']],
                'fuel_type': [optimization_result['fuel_type']]
            }
            
            df = pd.DataFrame(data)
            
            import os
            if os.path.exists(filename):
                existing_df = pd.read_csv(filename)
                df = pd.concat([existing_df, df], ignore_index=True)
            
            df.to_csv(filename, index=False)
            print(f"Optimization history saved to {filename}")
            
        except Exception as e:
            print(f"Error saving optimization history: {e}")
