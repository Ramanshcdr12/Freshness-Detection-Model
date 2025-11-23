"""
Main Streamlit application for Freshness Detection.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from image_processor import load_image_from_bytes
from freshness_analyzer import FreshnessAnalyzer
from heatmap_generator import HeatmapGenerator
from history_storage import HistoryStorage
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Freshness Detection",
    page_icon="🥬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = FreshnessAnalyzer()
if 'heatmap_gen' not in st.session_state:
    st.session_state.heatmap_gen = HeatmapGenerator()
if 'history' not in st.session_state:
    st.session_state.history = HistoryStorage()
if 'current_result' not in st.session_state:
    st.session_state.current_result = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .freshness-gauge {
        text-align: center;
        padding: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Sidebar navigation
    st.sidebar.title("🥬 Freshness Detection")
    page = st.sidebar.radio(
        "Navigation",
        ["Scan", "History", "About"]
    )
    
    if page == "Scan":
        scan_page()
    elif page == "History":
        history_page()
    elif page == "About":
        about_page()

def scan_page():
    """Main scan page."""
    st.markdown('<h1 class="main-header">Freshness Detection</h1>', unsafe_allow_html=True)
    st.markdown("### Upload an image or use your camera to scan fruits and vegetables")
    
    # Image input methods
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Camera")
        camera_image = st.camera_input("Take a photo", key="camera")
        if camera_image:
            st.session_state.uploaded_image = camera_image
    
    with col2:
        st.subheader("📁 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            key="uploader"
        )
        if uploaded_file:
            st.session_state.uploaded_image = uploaded_file
    
    # Display uploaded image
    if st.session_state.uploaded_image:
        try:
            image = Image.open(st.session_state.uploaded_image)
            st.image(image, caption="Image to analyze", use_container_width=True)
            
            # Scan button
            if st.button("🔍 Scan for Freshness", type="primary", use_container_width=True):
                analyze_image(st.session_state.uploaded_image)
        except Exception as e:
            st.error(f"Error displaying image: {str(e)}. Please try uploading a different image.")
            st.session_state.uploaded_image = None
    
    # Display results if available
    if st.session_state.current_result:
        display_results(st.session_state.current_result)

def analyze_image(image_input):
    """Analyze uploaded image."""
    with st.spinner("Analyzing image... Please wait."):
        try:
            import io
            
            # Handle different input types
            if isinstance(image_input, Image.Image):
                # PIL Image (from camera)
                img_bytes = io.BytesIO()
                image_input.save(img_bytes, format='PNG')
                img_bytes = img_bytes.getvalue()
            elif hasattr(image_input, 'read'):
                # Streamlit UploadedFile
                # Reset file pointer to beginning
                image_input.seek(0)
                img_bytes = image_input.read()
                # Reset again for potential future reads
                image_input.seek(0)
            elif isinstance(image_input, bytes):
                # Already bytes
                img_bytes = image_input
            else:
                st.error("Invalid image input. Please upload an image file or use the camera.")
                return
            
            if not img_bytes or len(img_bytes) == 0:
                st.error("Empty image file. Please try again with a valid image.")
                return
            
            # Try loading with OpenCV
            image = load_image_from_bytes(img_bytes)
            
            # If OpenCV fails, try using PIL to convert
            if image is None or image.size == 0:
                try:
                    # Try loading with PIL first, then convert to OpenCV format
                    pil_image = Image.open(io.BytesIO(img_bytes))
                    # Convert PIL to numpy array (RGB)
                    img_array = np.array(pil_image)
                    # Convert RGB to BGR for OpenCV
                    if len(img_array.shape) == 3:
                        if img_array.shape[2] == 4:  # RGBA
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                        else:  # RGB
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    image = img_array
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}. Please ensure the file is a valid image (JPG, PNG).")
                    return
            
            if image is None or image.size == 0 or len(image.shape) < 2:
                st.error("Error loading image. Please ensure the file is a valid image (JPG, PNG).")
                return
            
            # Check image dimensions
            if image.shape[0] < 50 or image.shape[1] < 50:
                st.warning("Image is very small. Results may be less accurate. Please use a higher resolution image.")
            
            # Analyze
            result = st.session_state.analyzer.analyze(image)
            
            if not result:
                st.error("Analysis failed. Please try again.")
                return
            
            # Generate heatmap
            try:
                heatmap, overlay = st.session_state.heatmap_gen.create_visualization(
                    image, result['features'], result['freshness_percentage']
                )
                result['heatmap_overlay'] = overlay
                result['heatmap'] = heatmap
            except Exception as e:
                st.warning(f"Could not generate heatmap: {str(e)}")
                result['heatmap_overlay'] = image
                result['heatmap'] = None
            
            result['original_image'] = image
            
            # Store result
            st.session_state.current_result = result
            
            st.success("Analysis complete!")
            
        except cv2.error as e:
            st.error(f"Image processing error: {str(e)}. Please ensure the image is valid.")
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

def display_results(result: dict):
    """Display analysis results."""
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    
    # Item information
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"### {result['item_display_name']}")
        if result.get('hindi_name'):
            st.markdown(f"*{result['hindi_name']}*")
    
    with col2:
        confidence = result['confidence'] * 100
        st.metric("Confidence", f"{confidence:.1f}%")
    
    # Freshness gauge
    st.markdown("### Freshness Level")
    freshness = result['freshness_percentage']
    
    # Create circular gauge using plotly
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = freshness,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Freshness"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': get_freshness_color(freshness)},
            'steps': [
                {'range': [0, 33], 'color': "lightgray"},
                {'range': [33, 66], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Storage information
    st.markdown("### 📅 Usable Days")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Without Refrigerator",
            f"{result['usable_days_ambient']} days",
            delta=None
        )
    
    with col2:
        st.metric(
            "With Refrigerator",
            f"{result['usable_days_fridge']} days",
            delta=None
        )
    
    # Nutrition information
    st.markdown("### 🥗 Nutrition Information")
    
    calories_info = result['calories']
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Calories", f"{calories_info['calories']:.0f}")
    
    with col2:
        st.metric("Fiber", f"{result.get('fiber_g', 0):.1f} g")
    
    with col3:
        vitamins_count = len(result.get('vitamins', []))
        st.metric("Vitamins", vitamins_count)
    
    # Vitamins and minerals
    if result.get('vitamins'):
        st.markdown("**Vitamins:** " + ", ".join(result['vitamins']))
    if result.get('minerals'):
        st.markdown("**Minerals:** " + ", ".join(result['minerals']))
    
    # Benefits
    st.markdown("### 💚 Health Benefits")
    for benefit in result.get('benefits', []):
        st.markdown(f"• {benefit}")
    
    # Action suggestion
    st.markdown("### 💡 Action Suggestion")
    action_color = get_action_color(result['freshness_percentage'])
    st.markdown(
        f'<div class="metric-card" style="background-color: {action_color}20; border-left: 4px solid {action_color};">'
        f'<strong>{result["action_suggestion"]}</strong></div>',
        unsafe_allow_html=True
    )
    
    # Heatmap visualization
    with st.expander("🔍 Why this result? (View Heatmap)"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Image**")
            st.image(result['original_image'], use_container_width=True, channels="BGR")
        
            with col2:
                st.markdown("**Freshness Heatmap**")
                if result.get('heatmap_overlay') is not None:
                    st.image(result['heatmap_overlay'], use_container_width=True, channels="BGR")
                else:
                    st.info("Heatmap not available for this image.")
        
        st.markdown(f"*{result['prediction_basis']}*")
        
        explanation = st.session_state.heatmap_gen.get_heatmap_explanation(
            result['heatmap'], result['freshness_percentage']
        )
        st.markdown(explanation)
    
    # Action buttons
    st.markdown("### Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💾 Save to History", use_container_width=True):
            save_to_history(result)
    
    with col2:
        if result.get('recipes'):
            if st.button("🍳 View Recipes", use_container_width=True):
                show_recipes(result['recipes'], result['item_display_name'])
    
    with col3:
        if st.button("📤 Share", use_container_width=True):
            st.info("Share functionality - Export results as image/PDF (to be implemented)")
    
    with col4:
        if st.button("🔄 Scan Another", use_container_width=True):
            st.session_state.current_result = None
            st.session_state.uploaded_image = None
            st.rerun()

def save_to_history(result: dict):
    """Save scan result to history."""
    try:
        # Prepare scan data
        scan_data = {
            'id': len(st.session_state.history.load_all_scans()) + 1,
            'timestamp': datetime.now().isoformat(),
            'item_name': result['item_name'],
            'item_display_name': result['item_display_name'],
            'freshness_percentage': result['freshness_percentage'],
            'confidence': result['confidence'],
            'usable_days_ambient': result['usable_days_ambient'],
            'usable_days_fridge': result['usable_days_fridge'],
            'calories': result['calories']['calories'],
            'action_suggestion': result['action_suggestion']
        }
        
        if st.session_state.history.save_scan(scan_data):
            st.success("Scan saved to history!")
        else:
            st.error("Error saving scan to history.")
    except Exception as e:
        st.error(f"Error saving scan: {str(e)}")

def show_recipes(recipes: list, item_name: str):
    """Display recipe suggestions."""
    st.markdown(f"### 🍳 Recipe Suggestions for {item_name}")
    for i, recipe in enumerate(recipes, 1):
        st.markdown(f"{i}. **{recipe}**")
    st.info("💡 Tip: These are traditional Indian recipes. Search online for detailed recipes!")

def history_page():
    """History page."""
    st.markdown("# 📚 Scan History")
    
    history = st.session_state.history.load_all_scans()
    
    if not history:
        st.info("No scan history yet. Start scanning to build your history!")
        return
    
    # Sort by timestamp (most recent first)
    history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    # Display history
    for scan in history:
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                timestamp = scan.get('timestamp', '')
                try:
                    dt = datetime.fromisoformat(timestamp)
                    date_str = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    date_str = timestamp
                
                st.markdown(f"**{scan.get('item_display_name', scan.get('item_name', 'Unknown'))}**")
                st.caption(date_str)
            
            with col2:
                freshness = scan.get('freshness_percentage', 0)
                st.metric("Freshness", f"{freshness:.0f}%")
            
            with col3:
                confidence = scan.get('confidence', 0) * 100
                st.metric("Confidence", f"{confidence:.0f}%")
            
            with col4:
                scan_id = scan.get('id')
                if st.button("🗑️ Delete", key=f"delete_{scan_id}"):
                    if st.session_state.history.delete_scan(scan_id):
                        st.success("Deleted!")
                        st.rerun()
            
            st.markdown(f"*{scan.get('action_suggestion', '')}*")
            st.markdown("---")
    
    # Clear all button
    if st.button("🗑️ Clear All History", type="secondary"):
        if st.session_state.history.clear_history():
            st.success("History cleared!")
            st.rerun()

def about_page():
    """About page."""
    st.markdown("# About Freshness Detection")
    
    st.markdown("""
    ## 🥬 Freshness Detection Web Application
    
    This application helps you determine the freshness of fruits and vegetables using 
    advanced image analysis and machine learning.
    
    ### Features:
    - **Freshness Detection**: Analyze freshness percentage (0-100%)
    - **Nutrition Information**: Get detailed nutrition facts, calories, vitamins, and benefits
    - **Storage Recommendations**: Learn how long items stay fresh with/without refrigeration
    - **Visual Analysis**: View heatmap visualizations showing areas affecting freshness decisions
    - **Scan History**: Save and review your previous scans
    
    ### How it works:
    1. Upload an image or use your camera to capture a fruit/vegetable
    2. Click "Scan for Freshness" to analyze
    3. View detailed results including freshness percentage, nutrition info, and recommendations
    4. Save scans to history for future reference
    
    ### Technology:
    - **Streamlit**: Web application framework
    - **OpenCV**: Image processing
    - **Scikit-learn**: Machine learning models
    - **Computer Vision**: Feature extraction and analysis
    
    ### Note:
    This application uses a combination of machine learning models and rule-based analysis
    to estimate freshness. Results are for informational purposes and should not be the
    sole basis for food safety decisions.
    """)

def get_freshness_color(freshness: float) -> str:
    """Get color based on freshness percentage."""
    if freshness >= 70:
        return "#4CAF50"  # Green
    elif freshness >= 40:
        return "#FF9800"  # Orange
    else:
        return "#F44336"  # Red

def get_action_color(freshness: float) -> str:
    """Get color for action suggestion."""
    if freshness >= 70:
        return "#4CAF50"  # Green
    elif freshness >= 40:
        return "#FF9800"  # Orange
    else:
        return "#F44336"  # Red

if __name__ == "__main__":
    main()

