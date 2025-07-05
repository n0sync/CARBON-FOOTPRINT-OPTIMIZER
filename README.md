# Carbon Footprint Optimizer - Supply Chain Logistics 

An AI-powered application that optimizes transportation routes for minimal environmental impact in supply chain logistics. Features a web-based dashboard built with Streamlit.


## Features 

* **Interactive Web Dashboard** with visual analytics
* **Route Optimization** based on carbon emissions, weather, and cargo
* **Multiple Route Alternatives** comparison (Eco-Friendly, Fastest, Balanced)
* **Deep Learning Model** for carbon footprint prediction
* **Sample Data Generation** for immediate experimentation
* **Interactive Maps** with Folium visualization
* **Performance Metrics** tracking emission reductions

## 🛠️ Tech Stack 

- **Frontend**: Streamlit, Plotly, Folium
- **Backend**: TensorFlow/Keras, Scikit-learn, Pandas
- **Mapping**: OSRM, GeoPy
- **Optimization**: Custom route optimization algorithms



## Requirements
- Python 3.8+
- Internet connection

## Setup Instructions

1. **Navigate to project directory:**
   ```bash
   cd CARBON-FOOTPRINT-OPTIMIZER
2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
4. **Run the Streamlit:**
   ```bash
   streamlit run main.py
5. **Open the Browser:**
   
   Visit: [http://localhost:8501](http://localhost:8501)



## Project Structure

```
CFO/
├── main.py              # Main Streamlit application entry 
├── requirements.txt         # Python dependencies
├── gui/
│   └── main_gui.py         # Streamlit interface
├── models/
│   └── carbon_model.py     # AI model 
├── utils/
│   └── route_optimizer.py  # Route optimization logic
├── data/
│   └── data_handler.py     # Data processing utilities
└── README.md               
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.


