# Carbon Footprint Optimizer - Supply Chain Logistics

A simple desktop app to help optimize delivery routes and reduce carbon emissions in supply chain logistics.

## Features

* Load sample logistics data (CSV)
* Train a deep learning model on the data
* Optimize routes based on cargo weight and weather
* User-friendly GUI built with Tkinter

## How to Run

1. Make sure Python is installed.
2. Create and activate a virtual environment (optional but recommended).
3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
python main.py
```

## File Structure

* `main.py`: Starts the GUI app
* `data/`: Handles CSV and sample data
* `models/`: Contains the training model
* `utils/`: Route optimization code
* `gui/`: GUI layout and event logic

## Notes

* Make sure `data/sample_logistics_data.csv` is present or it will be generated

---
Feel free to contribute or customize this app for your own logistics needs.
