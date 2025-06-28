import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
from gui.main_gui import MainGUI
from data.data_handler import DataHandler
from models.carbon_model import CarbonFootprintModel
from utils.route_optimizer import RouteOptimizer
import traceback
import os

class CarbonFootprintApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Carbon Footprint Optimizer - Supply Chain Logistics")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        self.data_handler = DataHandler()
        self.model = CarbonFootprintModel()
        self.route_optimizer = RouteOptimizer()
        self.gui = MainGUI(self.root, self)
        self.sample_data = None
        self.trained_model = None
        
    def load_sample_data(self):
        try:
            if not os.path.exists('data/sample_logistics_data.csv'):
                self.data_handler.generate_sample_data()
            self.sample_data = self.data_handler.load_data('data/sample_logistics_data.csv')
            messagebox.showinfo("Success", f"Loaded {len(self.sample_data)} records successfully!")
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            return False
    
    def train_model(self):
        if self.sample_data is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return False
        try:
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Training Model")
            progress_window.geometry("300x100")
            progress_label = tk.Label(progress_window, text="Training deep learning model...")
            progress_label.pack(pady=10)
            progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
            progress_bar.pack(pady=10)
            progress_bar.start()
            X, y = self.data_handler.prepare_training_data(self.sample_data)
            self.trained_model = self.model.train_model(X, y)
            progress_bar.stop()
            progress_window.destroy()
            messagebox.showinfo("Success", "Model trained successfully!")
            return True
        except Exception as e:
            if 'progress_window' in locals():
                progress_window.destroy()
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            return False
    
    def optimize_route(self, start_location, end_location, cargo_weight, weather_condition):
        if self.trained_model is None:
            messagebox.showwarning("Warning", "Please train the model first!")
            return None
        try:
            print(f"DEBUG: Calling with params: {start_location}, {end_location}, {cargo_weight}, {weather_condition}")
            optimized_route = self.route_optimizer.optimize_route(
                start_location=start_location, 
                end_location=end_location, 
                cargo_weight=cargo_weight, 
                weather_condition=weather_condition,
                trained_model=self.trained_model
            )
            print(f"DEBUG: Result: {optimized_route}")
            return optimized_route
        except Exception as e:
            print(f"DEBUG: Full error details: {e}")
            traceback.print_exc()
            messagebox.showerror("Error", f"Route optimization failed: {str(e)}")
            return None
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    if not os.path.exists('data'):
        os.makedirs('data')
    app = CarbonFootprintApp()
    app.run()
