
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime, timedelta
import random

class DataHandler:
    def __init__(self):
        self.label_encoders = {}
        self.onehot_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def generate_sample_data(self, num_records=1000, save=True):
        np.random.seed(42)
        random.seed(42)

        base_cities = {
            'Mumbai': (19.0760, 72.8777),
            'Delhi': (28.7041, 77.1025),
            'Bangalore': (12.9716, 77.5946),
            'Chennai': (13.0827, 80.2707),
            'Kolkata': (22.5726, 88.3639),
            'Hyderabad': (17.3850, 78.4867),
            'Pune': (18.5204, 73.8567),
            'Ahmedabad': (23.0225, 72.5714),
            'Jaipur': (26.9124, 75.7873),
            'Lucknow': (26.8467, 80.9462)
        }
        
        cities = base_cities.copy()
        cities.update({f"{city}, India": coords for city, coords in base_cities.items()})

        data = []
        start_date = datetime(2024, 1, 1)

        for i in range(num_records):
            start_city = random.choice(list(cities.keys()))
            end_city = random.choice([c for c in cities.keys() if c != start_city])

            start_coords = cities[start_city]
            end_coords = cities[end_city]
            distance = np.sqrt((start_coords[0] - end_coords[0]) ** 2 +
                            (start_coords[1] - end_coords[1]) ** 2) * 111
            distance = max(100, distance + np.random.normal(0, 50))

            cargo_weight = np.random.uniform(500, 5000)
            vehicle_type = random.choice(['Truck', 'Van', 'Container'])
            weather = random.choice(['Clear', 'Rainy', 'Cloudy', 'Foggy', 'Stormy'])
            traffic_density = random.choice(['Low', 'Medium', 'High'])
            fuel_type = random.choice(['Diesel', 'Petrol', 'CNG', 'Electric'])

            base_fuel = distance * 0.08 * (cargo_weight / 1000)
            weather_factor = {'Clear': 1.0, 'Rainy': 1.2, 'Cloudy': 1.05, 'Foggy': 1.15, 'Stormy': 1.3}[weather]
            traffic_factor = {'Low': 1.0, 'Medium': 1.15, 'High': 1.3}[traffic_density]
            weight_factor = 1 + (cargo_weight / 10000)

            fuel_consumption = base_fuel * (1 + np.random.normal(0, 0.15))

            emission_factor = {'Diesel': 2.68, 'Petrol': 2.31, 'CNG': 1.94, 'Electric': 0.5}[fuel_type]
            carbon_emissions = fuel_consumption * emission_factor
            carbon_emissions += np.random.normal(0, carbon_emissions * 0.1)
            carbon_emissions = max(10, carbon_emissions)

            date = start_date + timedelta(days=random.randint(0, 500))

            record = {
                'date': date.strftime('%Y-%m-%d %H:%M:%S'),
                'start_city': start_city,
                'end_city': end_city,
                'start_location': start_city,  
                'end_location': end_city,      
                'distance_km': round(distance, 2),
                'cargo_weight_kg': round(cargo_weight, 2),
                'vehicle_type': vehicle_type,
                'weather_condition': weather,
                'traffic_density': traffic_density,
                'fuel_type': fuel_type,
                'fuel_consumption_liters': round(fuel_consumption, 2),
                'carbon_emissions_kg': round(carbon_emissions, 2),
                'delivery_time_hours': round(distance / random.uniform(40, 80), 2),
                'cost_inr': round(distance * random.uniform(8, 15) + cargo_weight * 0.5, 2)
            }

            data.append(record)

        df = pd.DataFrame(data)

        if save:
            os.makedirs('data', exist_ok=True)
            df.to_csv('data/sample_logistics_data.csv', index=False)

        return df

    
    def load_data(self, file_path):
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded data with shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self, df):
        processed_df = df.copy()
        
        processed_df = processed_df.fillna(processed_df.mean(numeric_only=True))
        
        processed_df['date'] = pd.to_datetime(processed_df['date'])
        processed_df['month'] = processed_df['date'].dt.month
        processed_df['day_of_week'] = processed_df['date'].dt.dayofweek
        processed_df['quarter'] = processed_df['date'].dt.quarter
        
        categorical_columns = ['start_city', 'end_city', 'vehicle_type', 
                              'weather_condition', 'traffic_density', 'fuel_type']
        
        label_encode_cols = ['start_city', 'end_city', 'weather_condition', 'traffic_density']
        for col in label_encode_cols:
            if col in processed_df.columns:
                le = LabelEncoder()
                processed_df[f'{col}_encoded'] = le.fit_transform(processed_df[col])
                self.label_encoders[col] = le
        
        onehot_encode_cols = ['vehicle_type', 'fuel_type']
        for col in onehot_encode_cols:
            if col in processed_df.columns:
                dummies = pd.get_dummies(processed_df[col], prefix=col)
                processed_df = pd.concat([processed_df, dummies], axis=1)
        
        return processed_df
    
    def prepare_training_data(self, df):
        processed_df = self.preprocess_data(df)
        
        feature_columns = [
            'distance_km', 'cargo_weight_kg', 'fuel_consumption_liters',
            'delivery_time_hours', 'month', 'day_of_week', 'quarter',
            'start_city_encoded', 'end_city_encoded', 
            'weather_condition_encoded', 'traffic_density_encoded'
        ]
        
        onehot_columns = [col for col in processed_df.columns 
                         if col.startswith(('vehicle_type_', 'fuel_type_'))]
        feature_columns.extend(onehot_columns)
        
        feature_columns = [col for col in feature_columns if col in processed_df.columns]
        self.feature_columns = feature_columns
        
        X = processed_df[feature_columns]
        y = processed_df['carbon_emissions_kg']
        
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Training data shape: X={X_scaled.shape}, y={y.shape}")
        print(f"Feature columns: {feature_columns}")
        
        return X_scaled, y.values
    
    def prepare_prediction_data(self, start_city, end_city, distance, cargo_weight, 
                               weather, traffic, vehicle_type, fuel_type):
        data = {
            'distance_km': [distance],
            'cargo_weight_kg': [cargo_weight],
            'fuel_consumption_liters': [distance / 6.0],
            'delivery_time_hours': [distance / 60.0],
            'month': [6],
            'day_of_week': [2],
            'quarter': [2]
        }
        
        if 'start_city' in self.label_encoders:
            try:
                data['start_city_encoded'] = [self.label_encoders['start_city'].transform([start_city])[0]]
            except ValueError:
                data['start_city_encoded'] = [0]
        
        if 'end_city' in self.label_encoders:
            try:
                data['end_city_encoded'] = [self.label_encoders['end_city'].transform([end_city])[0]]
            except ValueError:
                data['end_city_encoded'] = [0]
        
        if 'weather_condition' in self.label_encoders:
            try:
                data['weather_condition_encoded'] = [self.label_encoders['weather_condition'].transform([weather])[0]]
            except ValueError:
                data['weather_condition_encoded'] = [0]
        
        if 'traffic_density' in self.label_encoders:
            try:
                data['traffic_density_encoded'] = [self.label_encoders['traffic_density'].transform([traffic])[0]]
            except ValueError:
                data['traffic_density_encoded'] = [0]
        
        vehicle_types = ['Truck', 'Van', 'Container']
        for vtype in vehicle_types:
            data[f'vehicle_type_{vtype}'] = [1 if vehicle_type == vtype else 0]
        
        fuel_types = ['Diesel', 'Petrol', 'CNG', 'Electric']
        for ftype in fuel_types:
            data[f'fuel_type_{ftype}'] = [1 if fuel_type == ftype else 0]
        
        pred_df = pd.DataFrame(data)
        
        for col in self.feature_columns:
            if col not in pred_df.columns:
                pred_df[col] = 0
        
        X_pred = pred_df[self.feature_columns]
        
        X_pred_scaled = self.scaler.transform(X_pred)
        
        return X_pred_scaled
    
    def get_data_statistics(self, df):
        stats = {
            'total_records': len(df),
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max()
            },
            'carbon_emissions': {
                'mean': df['carbon_emissions_kg'].mean(),
                'std': df['carbon_emissions_kg'].std(),
                'min': df['carbon_emissions_kg'].min(),
                'max': df['carbon_emissions_kg'].max()
            },
            'distance': {
                'mean': df['distance_km'].mean(),
                'std': df['distance_km'].std(),
                'min': df['distance_km'].min(),
                'max': df['distance_km'].max()
            },
            'cargo_weight': {
                'mean': df['cargo_weight_kg'].mean(),
                'std': df['cargo_weight_kg'].std(),
                'min': df['cargo_weight_kg'].min(),
                'max': df['cargo_weight_kg'].max()
            },
            'unique_cities': {
                'start_cities': df['start_city'].nunique(),
                'end_cities': df['end_city'].nunique()
            },
            'vehicle_types': df['vehicle_type'].value_counts().to_dict(),
            'weather_conditions': df['weather_condition'].value_counts().to_dict(),
            'fuel_types': df['fuel_type'].value_counts().to_dict()
        }
        
        return stats
