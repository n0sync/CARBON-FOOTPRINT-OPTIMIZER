import streamlit as st
import sys
import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from gui.main_gui import StreamlitCarbonFootprintGUI
    
    def main():
        try:
            app = StreamlitCarbonFootprintGUI()
            app.render_main_interface()
        except Exception as e:
            st.error(f"Application Error: {str(e)}")
            st.info("Please check your environment setup and try again.")
    
    if __name__ == "__main__":
        main()

except ImportError as e:
    st.error(f"Import Error: {str(e)}")
    st.info("Please ensure all required packages are installed. Run: pip install -r requirements.txt")
except Exception as e:
    st.error(f"Startup Error: {str(e)}")
    st.info("Please check your project structure and dependencies.")
