# =========================================================
# Dashboard V3 - Divorce Forecast Dashboard
# Run with: python -m streamlit run DashboardV3.py
# =========================================================

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from prophet import Prophet
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# Page Configuration (Must be first Streamlit command)
# =========================================================
st.set_page_config(
    page_title="Divorce Forecast Dashboard",
    page_icon="💍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# Constants
# =========================================================
DATA_FILES = {
    "divorce_model": "divorce_all_model.csv",
    "regional": "monthly_marriage_divorce_wide_BE.csv",
    "sarimax_metrics": "sarimax_metrics.csv",
    "sarimax_rolling": "sarimax_rolling_forecast.csv",
    "prophet_future": "prophet_forecast_future.csv",
    "sarimax_future": "sarima_rolling_future_forecast.csv",
    # "scenario": "TestScenarioPred.csv"
}

# Enhanced color scheme optimized for white backgrounds with semantic meaning
COLORS = {
    # === Core Teams ===
    "prophet":  "#1F77B4",   # 🔵 Prophet - Blue tone
    "sarimax":  "#C0392B",   # 🔴 SARIMAX - Red tone
    
    # === Observed Data ===
    "marriage": "#4FA3D1",   # 🔵 Marriage - Light blue
    "divorce":  "#E74C3C",   # 🔴 Divorce - Bright red
    "actual":   "#2C2C2C",   # ⚫ Neutral actual - Dark grey
    
    # === Simulation / Scenario ===
    "simulated": "#F39C12",  # 🟠 Amber - Simulation data
    
    # === UI Semantic ===
    "primary":   "#1F77B4",  # Primary blue
    "secondary": "#C0392B",  # Secondary red
    "success":   "#2ECC71",  # Success green
    "warning":   "#F39C12",  # Warning amber
    "danger":    "#E74C3C",  # Danger red
    "info":      "#3498DB"   # Info light blue
}

# Custom CSS for beautiful dashboard styling
CUSTOM_CSS = """
<style>
    /* Main container styling */
    .main {
        background-color: #FFFFFF;
    }
    
    /* Header styling */
    h1 {
        color: #2C3E50;
        font-weight: 700;
        padding-bottom: 10px;
        border-bottom: 3px solid #3498DB;
    }
    
    h2, h3 {
        color: #34495E;
        font-weight: 600;
    }
    
    /* Metric cards styling */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #2C3E50;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 15px;
        font-weight: 500;
        color: #7F8C8D;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #F8F9FA;
        border-right: 2px solid #E1E8ED;
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #2C3E50;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(50, 50, 93, 0.11), 0 1px 3px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 14px rgba(50, 50, 93, 0.1), 0 3px 6px rgba(0, 0, 0, 0.08);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F8F9FA;
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: white;
        border-radius: 8px;
        color: #5A6C7D;
        font-weight: 600;
        border: 2px solid #E1E8ED;
        padding: 0 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-color: #667eea;
    }
    
    /* Dataframe styling */
    .dataframe {
        border: 2px solid #E1E8ED !important;
        border-radius: 8px;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid #3498DB;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 2px solid #E1E8ED;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #F8F9FA;
        border-radius: 8px;
        font-weight: 600;
        color: #2C3E50;
    }
    
    /* Select boxes */
    div[data-baseweb="select"] {
        border-radius: 8px;
    }
    
    /* Slider */
    .stSlider {
        padding: 10px 0;
    }
</style>
"""

# =========================================================
# Region Schemes
# =========================================================
REGION_SCHEMES: Dict[str, Dict[str, List[str]]] = {
    "การแบ่งแบบสี่ภูมิภาค (กรมทางหลวง)": {
        "ภาคเหนือ": [
            "จังหวัดเชียงราย", "จังหวัดน่าน", "จังหวัดพะเยา", "จังหวัดเชียงใหม่",
            "จังหวัดแม่ฮ่องสอน", "จังหวัดแพร่", "จังหวัดลำปาง", "จังหวัดลำพูน",
            "จังหวัดอุตรดิตถ์", "จังหวัดพิษณุโลก", "จังหวัดสุโขทัย",
            "จังหวัดกำแพงเพชร", "จังหวัดนครสวรรค์", "จังหวัดพิจิตร"
        ],
        "ภาคตะวันออกเฉียงเหนือ": [
            "จังหวัดเพชรบูรณ์", "จังหวัดหนองคาย", "จังหวัดนครพนม", "จังหวัดสกลนคร",
            "จังหวัดอุดรธานี", "จังหวัดหนองบัวลำภู", "จังหวัดเลย", "จังหวัดมุกดาหาร",
            "จังหวัดกาฬสินธุ์", "จังหวัดขอนแก่น", "จังหวัดอำนาจเจริญ", "จังหวัดยโสธร",
            "จังหวัดร้อยเอ็ด", "จังหวัดมหาสารคาม", "จังหวัดชัยภูมิ",
            "จังหวัดนครราชสีมา", "จังหวัดบุรีรัมย์", "จังหวัดสุรินทร์",
            "จังหวัดศรีสะเกษ", "จังหวัดอุบลราชธานี", "จังหวัดบึงกาฬ"
        ],
        "ภาคกลาง": [
            "กรุงเทพมหานคร", "จังหวัดนนทบุรี", "จังหวัดปทุมธานี", "จังหวัดสมุทรปราการ",
            "จังหวัดสมุทรสาคร", "จังหวัดสมุทรสงคราม", "จังหวัดนครปฐม",
            "จังหวัดสุพรรณบุรี"
        ],
        "ภาคใต้": [
            "จังหวัดชุมพร", "จังหวัดระนอง", "จังหวัดสุราษฎร์ธานี",
            "จังหวัดนครศรีธรรมราช", "จังหวัดกระบี่", "จังหวัดพังงา", "จังหวัดภูเก็ต",
            "จังหวัดพัทลุง", "จังหวัดตรัง", "จังหวัดปัตตานี", "จังหวัดสงขลา",
            "จังหวัดสตูล", "จังหวัดนราธิวาส", "จังหวัดยะลา"
        ],
    },
    "การแบ่งอย่างเป็นทางการ (6 ภูมิภาค)": {
        "ภาคเหนือ": [
            "จังหวัดเชียงราย", "จังหวัดน่าน", "จังหวัดพะเยา", "จังหวัดเชียงใหม่",
            "จังหวัดแม่ฮ่องสอน", "จังหวัดแพร่", "จังหวัดลำปาง",
            "จังหวัดลำพูน", "จังหวัดอุตรดิตถ์"
        ],
        "ภาคตะวันออกเฉียงเหนือ": [
            "จังหวัดหนองคาย", "จังหวัดนครพนม", "จังหวัดสกลนคร", "จังหวัดอุดรธานี",
            "จังหวัดหนองบัวลำภู", "จังหวัดเลย", "จังหวัดมุกดาหาร", "จังหวัดกาฬสินธุ์",
            "จังหวัดขอนแก่น", "จังหวัดอำนาจเจริญ", "จังหวัดยโสธร", "จังหวัดร้อยเอ็ด",
            "จังหวัดมหาสารคาม", "จังหวัดชัยภูมิ", "จังหวัดนครราชสีมา",
            "จังหวัดบุรีรัมย์", "จังหวัดสุรินทร์", "จังหวัดศรีสะเกษ",
            "จังหวัดอุบลราชธานี", "จังหวัดบึงกาฬ"
        ],
        "ภาคตะวันตก": [
            "จังหวัดตาก", "จังหวัดกาญจนบุรี", "จังหวัดราชบุรี", "จังหวัดเพชรบุรี", "จังหวัดประจวบคีรีขันธ์"
        ],
        "ภาคกลาง": [
            "กรุงเทพมหานคร", "จังหวัดปทุมธานี", "จังหวัดนนทบุรี",
            "จังหวัดนครปฐม", "จังหวัดสมุทรปราการ"
        ],
        "ภาคตะวันออก": [
            "จังหวัดชลบุรี", "จังหวัดระยอง", "จังหวัดจันทบุรี", "จังหวัดตราด"
        ],
        "ภาคใต้": [
            "จังหวัดชุมพร", "จังหวัดสุราษฎร์ธานี", "จังหวัดนครศรีธรรมราช",
            "จังหวัดสงขลา", "จังหวัดยะลา", "จังหวัดนราธิวาส"
        ],
    }
}

# =========================================================
# Data Loading Functions with Enhanced Caching
# =========================================================

@st.cache_data(show_spinner="Loading divorce model data...")
def load_data() -> pd.DataFrame:
    """Load and preprocess the divorce and marriage data for model comparison"""
    try:
        df = pd.read_csv(DATA_FILES["divorce_model"])
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values("ds")
        return df
    except FileNotFoundError:
        st.error(f"❌ File not found: {DATA_FILES['divorce_model']}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading divorce model data: {str(e)}")
        return pd.DataFrame()


@st.cache_data(show_spinner="Loading regional data...")
def load_regional_data() -> pd.DataFrame:
    """Load regional data for province/region filtering"""
    try:
        df = pd.read_csv(DATA_FILES["regional"])
        return df
    except FileNotFoundError:
        st.error(f"❌ File not found: {DATA_FILES['regional']}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading regional data: {str(e)}")
        return pd.DataFrame()


# =========================================================
# Prophet Model Functions (from basic_Prophet.ipynb)
# =========================================================

@st.cache_data(show_spinner="Training Prophet model with hyperparameter tuning...")
def train_prophet_model(df: pd.DataFrame) -> Tuple[Prophet, pd.DataFrame, Dict]:
    """
    Train Prophet model with optimized parameters from basic_Prophet.ipynb
    Returns: (trained_model, prepared_dataframe, parameters_used)
    """
    # Prepare data
    df_prophet = df[['ds', 'Divorce']].copy()
    df_prophet = df_prophet.rename(columns={"Divorce": "y"})
    
    # Set cap/floor for logistic growth
    global_cap = df_prophet['y'].max() * 1.2
    global_floor = 0
    df_prophet['cap'] = global_cap
    df_prophet['floor'] = global_floor
    
    # Use best parameters from basic_Prophet.ipynb
    # These were found through extensive grid search
    best_params = {
        'changepoint_prior_scale': 1.0,
        'seasonality_prior_scale': 0.1,
        'yearly_seasonality': True,
        'weekly_seasonality': False,
        'daily_seasonality': False
    }
    
    # Train Prophet model with logistic growth
    model = Prophet(growth='logistic', **best_params)
    model.fit(df_prophet)
    
    return model, df_prophet, best_params


@st.cache_data(show_spinner="Calculating Prophet metrics...")
def calculate_prophet_metrics_from_forecast(_model, df: pd.DataFrame) -> Dict:
    """
    Calculate Prophet metrics by comparing forecast with actual data
    Based on the approach from basic_Prophet.ipynb
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    
    # Get forecast for the training period
    df_prophet = df[['ds', 'Divorce']].copy()
    df_prophet = df_prophet.rename(columns={"Divorce": "y"})
    
    # Set cap/floor
    global_cap = df_prophet['y'].max() * 1.2
    global_floor = 0
    df_prophet['cap'] = global_cap
    df_prophet['floor'] = global_floor
    
    # Generate forecast
    future = _model.make_future_dataframe(periods=0, freq='MS')  # Only historical period
    future['cap'] = global_cap
    future['floor'] = global_floor
    forecast = _model.predict(future)
    
    # Merge forecast with actual data
    metric_df = (
        forecast[['ds', 'yhat']]
        .merge(df_prophet[['ds', 'y']], on='ds', how='inner')
    )
    
    y_true = metric_df['y']
    y_pred = metric_df['yhat']
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'forecast_df': metric_df.rename(columns={'y': 'Actual', 'yhat': 'Forecast'})
    }


@st.cache_data(show_spinner="Generating Prophet future forecast...")
def generate_prophet_future_forecast(_model, df_prophet: pd.DataFrame, periods: int = 60) -> pd.DataFrame:
    """
    Generate future forecast using trained Prophet model
    Args:
        _model: Trained Prophet model (underscore prefix to skip caching this arg)
        df_prophet: Prepared dataframe with cap/floor
        periods: Number of periods to forecast (default: 60 months = 5 years)
    """
    # Create future dataframe
    future = _model.make_future_dataframe(periods=periods, freq='MS')
    future['cap'] = df_prophet['cap'].iloc[0]
    future['floor'] = df_prophet['floor'].iloc[0]
    
    # Generate forecast
    forecast = _model.predict(future)
    
    # Return relevant columns
    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']].copy()
    return result


@st.cache_data(show_spinner="Loading model metrics...")
def load_metrics() -> pd.DataFrame:
    """Load model metrics for Prophet and SARIMAX"""
    try:
        prophet = pd.read_csv(DATA_FILES["prophet_metrics"])
        arima = pd.read_csv(DATA_FILES["sarimax_metrics"])

        # Add model identifier
        prophet["Model"] = "Prophet"
        arima["Model"] = "SARIMAX"

        # Normalize column names to uppercase
        prophet.columns = prophet.columns.str.upper()
        arima.columns = arima.columns.str.upper()

        # Select consistent columns from both
        common_cols = ["MODEL", "ROUND", "TRAIN", "TEST", "MAE", "RMSE", "MAPE"]
        prophet = prophet[common_cols]
        arima = arima[common_cols]

        return pd.concat([prophet, arima], ignore_index=True)
    except FileNotFoundError as e:
        st.error(f"❌ Metrics file not found: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading metrics: {str(e)}")
        return pd.DataFrame()



@st.cache_data(show_spinner="Loading SARIMAX rolling forecast...")
def load_sarimax_rolling() -> pd.DataFrame:
    """Load SARIMAX rolling forecast data only"""
    try:
        arima_roll = pd.read_csv(DATA_FILES["sarimax_rolling"])
        arima_roll["ds"] = pd.to_datetime(arima_roll["ds"])
        return arima_roll
    except FileNotFoundError as e:
        st.error(f"❌ SARIMAX rolling forecast file not found: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading SARIMAX rolling forecast: {str(e)}")
        return pd.DataFrame()


@st.cache_data(show_spinner="Loading future forecasts...")
def load_future_forecasts() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load future forecast data"""
    try:
        prophet_future = pd.read_csv(DATA_FILES["prophet_future"])
        arima_future = pd.read_csv(DATA_FILES["sarimax_future"])
        
        prophet_future["ds"] = pd.to_datetime(prophet_future["ds"])
        arima_future["ds"] = pd.to_datetime(arima_future["ds"])
        
        return prophet_future, arima_future
    except FileNotFoundError as e:
        st.error(f"❌ Future forecast file not found: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading future forecasts: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()


# =========================================================
# Helper Functions
# =========================================================

def create_province_to_region_mapping(scheme: str) -> Dict[str, str]:
    """Create a mapping from province names to region names"""
    mapping = {}
    for region, provinces in REGION_SCHEMES[scheme].items():
        for province in provinces:
            mapping[province] = region
    return mapping


def calculate_divorce_rate(marriages: float, divorces: float) -> float:
    """Calculate divorce rate as percentage"""
    return (divorces / marriages * 100) if marriages > 0 else 0.0


def format_number(number: float, decimals: int = 0) -> str:
    """Format number with thousands separator"""
    return f"{number:,.{decimals}f}"


# # =========================================================
# # Scenario Testing Functions
# # =========================================================

# def get_monthly_stats(df_scenario: pd.DataFrame) -> pd.DataFrame:
#     """
#     Calculate Mean, Std, Min, Max by month (1-12)
#     to use as bounds for random simulation
#     """
#     if 'month' not in df_scenario.columns:
#         df_scenario['month'] = df_scenario['ds'].dt.month
        
#     monthly_stats = df_scenario.groupby('month')['y'].agg(
#         avg='mean',
#         std='std',
#         floor_min=lambda x: x.quantile(0.2),
#         cap_max=lambda x: x.quantile(0.8)
#     ).reset_index()
    
#     return monthly_stats


# def generate_random_scenario(
#     monthly_stats: pd.DataFrame, 
#     start_date: pd.Timestamp, 
#     sim_years: int
# ) -> pd.DataFrame:
#     """
#     Generate random scenario data based on monthly statistics
#     Uses normal distribution with mean/std and clips by min/max
#     """
#     future_rows = []
    
#     for i in range(1, sim_years + 1):
#         for month in range(1, 13):
#             stats = monthly_stats[monthly_stats['month'] == month].iloc[0]
            
#             # Random value using normal distribution
#             rand_val = np.random.normal(stats['avg'], stats['std'])
            
#             # Clip to stay within bounds
#             final_val = np.clip(rand_val, stats['floor_min'], stats['cap_max'])
            
#             # Calculate date
#             current_date = start_date + pd.DateOffset(years=(i-1), months=month)
            
#             future_rows.append({
#                 'ds': current_date,
#                 'y': final_val
#             })
            
#     return pd.DataFrame(future_rows)


# def tune_prophet_hyperparameters(
#     df_train: pd.DataFrame, 
#     global_cap: float, 
#     global_floor: float,
#     n_samples: int = 20
# ) -> Dict:
#     """
#     Perform hyperparameter tuning for Prophet model using cross-validation
#     Returns the best parameters found
#     """
#     from itertools import product
#     import random
    
#     # Define parameter grid (simplified for dashboard usage)
#     param_grid = {
#         'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
#         'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
#         "yearly_seasonality": [True],
#         "weekly_seasonality": [False],
#         "daily_seasonality": [False]
#     }
    
#     # Generate all combinations
#     keys = param_grid.keys()
#     values = param_grid.values()
#     param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    
#     best_params = None
#     best_mape = float('inf')
    
#     # Sample combinations for faster tuning
#     random.seed(42)
#     sampled_combinations = random.sample(
#         param_combinations, 
#         min(n_samples, len(param_combinations))
#     )
    
#     progress_bar = st.progress(0)
#     status_text = st.empty()
    
#     for idx, params in enumerate(sampled_combinations):
#         try:
#             status_text.text(f"Testing parameter combination {idx + 1}/{len(sampled_combinations)}...")
            
#             # Split data for validation
#             train_size = int(len(df_train) * 0.8)
#             df_train_subset = df_train.iloc[:train_size].copy()
#             df_val = df_train.iloc[train_size:].copy()
            
#             # Train model with current parameters
#             m_val = Prophet(growth='logistic', **params)
#             m_val.fit(df_train_subset)
            
#             # Predict on validation set
#             future_val = m_val.make_future_dataframe(periods=len(df_val), freq='MS')
#             future_val['cap'] = global_cap
#             future_val['floor'] = global_floor
            
#             forecast_val = m_val.predict(future_val)
            
#             # Calculate MAPE on validation set
#             y_true = df_val['y'].values
#             y_pred = forecast_val['yhat'].tail(len(df_val)).values
            
#             # Avoid division by zero
#             mask = y_true != 0
#             if mask.sum() > 0:
#                 mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
#             else:
#                 mape = float('inf')
            
#             # Update best parameters if this is better
#             if mape < best_mape:
#                 best_mape = mape
#                 best_params = params.copy()
            
#             progress_bar.progress((idx + 1) / len(sampled_combinations))
            
#         except Exception as e:
#             # Skip this combination if it fails
#             continue
    
#     progress_bar.empty()
#     status_text.empty()
    
#     # Return best params or default if tuning failed
#     if best_params is None:
#         best_params = {
#             'changepoint_prior_scale': 0.05,
#             'seasonality_prior_scale': 10.0,
#             'yearly_seasonality': True,
#             'weekly_seasonality': False,
#             'daily_seasonality': False
#         }
#         st.warning("⚠️ Hyperparameter tuning failed, using default parameters")
#     else:
#         st.success(f"✨ Best parameters found! Validation MAPE: {best_mape:.4f}%")
    
#     return best_params


# =========================================================
# Load All Data
# =========================================================

try:
    # Load base data
    df = load_data()
    df_regional = load_regional_data()
    
    # Train Prophet model and generate forecasts (live from basic_Prophet.ipynb)
    prophet_model, df_prophet_prepared, prophet_params = train_prophet_model(df)
    prophet_metrics_dict = calculate_prophet_metrics_from_forecast(prophet_model, df)
    prophet_future = generate_prophet_future_forecast(prophet_model, df_prophet_prepared, periods=60)
    
    # Load SARIMAX data from CSV files
    arima_metrics = pd.read_csv(DATA_FILES["sarimax_metrics"])
    arima_metrics["Model"] = "SARIMAX"
    arima_metrics.columns = arima_metrics.columns.str.upper()
    metrics_df = arima_metrics  # Only SARIMAX metrics in the table
    
    arima_roll = pd.read_csv(DATA_FILES["sarimax_rolling"])
    arima_roll["ds"] = pd.to_datetime(arima_roll["ds"])
    
    arima_future = pd.read_csv(DATA_FILES["sarimax_future"])
    arima_future["ds"] = pd.to_datetime(arima_future["ds"])
    
    # Check if data loaded successfully
    if df.empty or df_regional.empty:
        st.error("❌ Failed to load required data files. Please check file paths.")
        st.stop()
        
except Exception as e:
    st.error(f"❌ Critical error loading data: {str(e)}")
    st.stop()


# =========================================================
# Apply Custom CSS
# =========================================================
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================================================
# Header
# =========================================================

st.image("Divorce_Banner.jpg", use_container_width=True)
st.markdown(f"*Model Comparison*")

# =========================================================
# Sidebar Filters
# =========================================================

st.sidebar.header("🔎 Filters & Settings")

# Reset Button
if st.sidebar.button("🔄 Reset to Default Filters", use_container_width=True):
    st.rerun()

# Region Scheme Selection
scheme = st.sidebar.selectbox(
    "1️⃣ รูปแบบการแบ่งภูมิภาค",
    list(REGION_SCHEMES.keys()),
    index=0,
    help="เลือกรูปแบบการแบ่งภูมิภาคที่ต้องการวิเคราะห์"
)

# Region Selection
region_options = ["ทั้งหมด"] + list(REGION_SCHEMES[scheme].keys())
region = st.sidebar.selectbox(
    "2️⃣ เลือกภูมิภาค", 
    region_options, 
    index=0,
    help="เลือกภูมิภาคที่ต้องการวิเคราะห์"
)

# Province Selection
province_options = ["ทั้งหมด"]
if region == "ทั้งหมด":
    for provs in REGION_SCHEMES[scheme].values():
        province_options.extend(provs)
else:
    province_options.extend(REGION_SCHEMES[scheme][region])

province_options = ["ทั้งหมด"] + sorted(set(province_options) - {"ทั้งหมด"})
province = st.sidebar.selectbox(
    "3️⃣ เลือกจังหวัด", 
    province_options, 
    index=0,
    help="เลือกจังหวัดที่ต้องการวิเคราะห์โดยเฉพาะ"
)

# Year Range Selection
if not df_regional.empty and "Year_BE" in df_regional.columns:
    year_min = int(df_regional.Year_BE.min())
    year_max = int(df_regional.Year_BE.max())
    year_range = st.sidebar.slider(
        "ช่วงปี (พ.ศ.)", 
        year_min, 
        year_max, 
        (year_min, year_max),
        help="เลือกช่วงปีที่ต้องการวิเคราะห์"
    )
else:
    year_range = (2560, 2565)

st.sidebar.divider()

# Model Selection (Fixed to both models for)
models_to_show = ["Prophet", "SARIMAX"]
st.sidebar.info("**📊 Models:** Prophet & SARIMAX")

# Display Options
st.sidebar.subheader("⚙️ Display Options")
show_confidence_intervals = st.sidebar.checkbox(
    "Show Confidence Intervals",
    value=True,
    help="Display prediction confidence intervals in charts"
)

show_data_tables = st.sidebar.checkbox(
    "Show Data Tables",
    value=False,
    help="Display raw data tables below charts"
)

# =========================================================
# Apply Filters to Regional Data
# =========================================================

df_filt = df_regional[
    (df_regional.Year_BE >= year_range[0]) & 
    (df_regional.Year_BE <= year_range[1])
].copy()

if province != "ทั้งหมด":
    df_filt = df_filt[df_filt.Province == province]
elif region != "ทั้งหมด":
    df_filt = df_filt[df_filt.Province.isin(REGION_SCHEMES[scheme][region])]

# =========================================================
# KPI Metrics
# =========================================================

total_marriages = df_filt.Marriage.sum()
total_divorces = df_filt.Divorce.sum()
divorce_rate = calculate_divorce_rate(total_marriages, total_divorces)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "💍 Total Marriages", 
        format_number(total_marriages),
        help="Total number of marriages in selected period/region"
    )

with col2:
    st.metric(
        "💔 Total Divorces", 
        format_number(total_divorces),
        help="Total number of divorces in selected period/region"
    )

with col3:
    st.metric(
        "📉 Divorce Rate", 
        f"{divorce_rate:.2f}%",
        help="Percentage of divorces relative to marriages"
    )

with col4:
    if not df_filt.empty and "Year_BE" in df_filt.columns:
        years_analyzed = df_filt.Year_BE.nunique()
        st.metric(
            "📅 Years Analyzed", 
            years_analyzed,
            help="Number of unique years in the filtered dataset"
        )

st.divider()

# =========================================================
# Trend Chart
# =========================================================

st.subheader("📈 Marriage vs Divorce Trend")

df_year = df_filt.groupby("Year_BE").sum().reset_index()

fig_trend = go.Figure()

fig_trend.add_trace(go.Scatter(
    x=df_year.Year_BE, 
    y=df_year.Marriage, 
    name="Marriage",
    line=dict(color=COLORS["marriage"], width=3),
    mode='lines+markers'
))

fig_trend.add_trace(go.Scatter(
    x=df_year.Year_BE, 
    y=df_year.Divorce, 
    name="Divorce",
    line=dict(color=COLORS["divorce"], width=3),
    mode='lines+markers'
))

fig_trend.update_layout(
    title={
        'text': "Marriage vs Divorce Trend Over Time",
        'font': {'size': 22, 'color': '#2C3E50', 'family': 'Arial Black'}
    },
    xaxis_title="Year (พ.ศ.)",
    yaxis_title="Count",
    hovermode="x unified",
    plot_bgcolor='rgba(240, 242, 245, 0.8)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Arial, sans-serif", size=12, color="#2C3E50"),
    template="plotly_white",
    height=450
)

st.plotly_chart(fig_trend, use_container_width=True)

st.divider()

# =========================================================
# Top Provinces - Pie Charts
# =========================================================

st.subheader("🏆 Top 5 Provinces")

col1, col2 = st.columns(2)

# Top Divorce Provinces
top_divorce = (
    df_filt.groupby("Province")["Divorce"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .reset_index()
)

# Top Marriage Provinces
top_marriage = (
    df_filt.groupby("Province")["Marriage"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .reset_index()
)

with col1:
    # Custom color gradient for Marriage: #FFF2E0 (lowest) to #898AC4 (highest)
    marriage_colors = ["#1F2287", "#393CA4", "#5557B6", "#7D7FCF", "#B3B4E8"]

    fig_pie_m = px.pie(
        top_marriage, 
        names="Province", 
        values="Marriage",
        title="💍 Top 5 Marriage Provinces",
        color_discrete_sequence=marriage_colors,
        hole=0.3  # Makes it a donut chart for modern look
    )
    fig_pie_m.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        textfont_size=13,
        marker=dict(line=dict(color='white', width=2))
    )
    fig_pie_m.update_layout(
        font=dict(family="Arial, sans-serif", size=12, color="#2C3E50"),
        title={
            'font': {'size': 18, 'color': '#2C3E50', 'family': 'Arial Black'}
        },
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_pie_m, use_container_width=True)

with col2:
    # Custom color gradient for Divorce: #FEEAC9 (lowest) to #FD7979 (highest)
    divorce_colors = ["#BA0E0E", "#ED2424", "#FF6D6D", "#FF9696", "#FFB8B8"]
    fig_pie_d = px.pie(
        top_divorce, 
        names="Province", 
        values="Divorce",
        title="💔 Top 5 Divorce Provinces",
        color_discrete_sequence=divorce_colors,
        hole=0.3  # Makes it a donut chart for modern look
    )
    fig_pie_d.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        textfont_size=13,
        marker=dict(line=dict(color='white', width=2))
    )
    fig_pie_d.update_layout(
        font=dict(family="Arial, sans-serif", size=12, color="#2C3E50"),
        title={
            'font': {'size': 18, 'color': '#2C3E50', 'family': 'Arial Black'}
        },
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_pie_d, use_container_width=True)

st.divider()

# =========================================================
# Regional Marriage Rate Ranking
# =========================================================

st.subheader("💍 Regional Marriage Rate Ranking")

# Create province to region mapping
province_to_region = create_province_to_region_mapping(scheme)

df_region_rank_marriage = df_filt.copy()
df_region_rank_marriage["Region"] = df_region_rank_marriage["Province"].map(province_to_region)

# Remove unmapped provinces
df_region_rank_marriage = df_region_rank_marriage.dropna(subset=["Region"])

# Aggregate by region
df_region_rank_marriage = (
    df_region_rank_marriage
    .groupby("Region")
    .agg(
        Marriage=("Marriage", "sum")
    )
    .reset_index()
)

# Calculate marriage rate as percentage of total marriages
total_marriages_all = df_region_rank_marriage["Marriage"].sum()
df_region_rank_marriage["Marriage_Rate"] = (
    df_region_rank_marriage["Marriage"] / total_marriages_all * 100
)

df_region_rank_marriage = df_region_rank_marriage.sort_values("Marriage_Rate", ascending=False)

# Create bar chart with custom color gradient: #FFF2E0 (lowest) to #898AC4 (highest)
fig_region_rank_marriage = px.bar(
    df_region_rank_marriage,
    x="Region",
    y="Marriage_Rate",
    text=df_region_rank_marriage["Marriage_Rate"].round(2),
    title="💍 Marriage Rate (%) by Region",
    color="Marriage_Rate",
    color_continuous_scale=["#B3B4E8", "#7D7FCF", "#5557B6", "#393CA4", "#1F2287"] # Custom gradient
)

fig_region_rank_marriage.update_traces(textposition="outside", texttemplate='%{text:.2f}%')
fig_region_rank_marriage.update_layout(
    title={
        'text': "💍 Marriage Rate (%) by Region",
        'font': {'size': 20, 'color': '#2C3E50', 'family': 'Arial Black'}
    },
    xaxis_title="Region",
    yaxis_title="Marriage Rate (%)",
    yaxis_tickformat=".2f",
    template="plotly_white",
    plot_bgcolor='rgba(240, 242, 245, 0.8)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Arial, sans-serif", size=12, color="#2C3E50"),
    height=400,
    showlegend=False
)

st.plotly_chart(fig_region_rank_marriage, use_container_width=True)

st.divider()

# =========================================================
# Regional Divorce Rate Ranking
# =========================================================

st.subheader("📊 Regional Divorce Rate Ranking")

# Create province to region mapping
province_to_region = create_province_to_region_mapping(scheme)

df_region_rank = df_filt.copy()
df_region_rank["Region"] = df_region_rank["Province"].map(province_to_region)

# Remove unmapped provinces
df_region_rank = df_region_rank.dropna(subset=["Region"])

# Aggregate by region
df_region_rank = (
    df_region_rank
    .groupby("Region")
    .agg(
        Marriage=("Marriage", "sum"),
        Divorce=("Divorce", "sum")
    )
    .reset_index()
)

df_region_rank["Divorce_Rate"] = (
    df_region_rank["Divorce"] / df_region_rank["Marriage"] * 100
)

df_region_rank = df_region_rank.sort_values("Divorce_Rate", ascending=False)

# Create bar chart with custom color gradient: #E6D9A2 (lowest) to #624E88 (highest)
fig_region_rank = px.bar(
    df_region_rank,
    x="Region",
    y="Divorce_Rate",
    text=df_region_rank["Divorce_Rate"].round(2),
    title="📉 Divorce Rate (%) by Region",
    color="Divorce_Rate",
    color_continuous_scale=["#FFB8B8", "#FF9696", "#FF6D6D", "#ED2424", "#BA0E0E"]  # Custom gradient
)

fig_region_rank.update_traces(textposition="outside", texttemplate='%{text:.2f}%')
fig_region_rank.update_layout(
    title={
        'text': "📉 Divorce Rate (%) by Region",
        'font': {'size': 20, 'color': '#2C3E50', 'family': 'Arial Black'}
    },
    xaxis_title="Region",
    yaxis_title="Divorce Rate (%)",
    yaxis_tickformat=".2f",
    template="plotly_white",
    plot_bgcolor='rgba(240, 242, 245, 0.8)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Arial, sans-serif", size=12, color="#2C3E50"),
    height=400,
    showlegend=False
)

st.plotly_chart(fig_region_rank, use_container_width=True)

st.divider()

# =========================================================
# Tabbed Interface for Model Analysis
# =========================================================

# tab1, tab2, tab3, tab4 = st.tabs([
#     "📊 Model Metrics",
#     "📉 Rolling Forecast",
#     "🔮 Future Forecast",
#     "🧪 Prophet Scenario Test"
# ])
tab1, tab2, tab3 = st.tabs([
    "📊 Model Metrics",
    "📉 Rolling Forecast",
    "🔮 Future Forecast"
])
# =========================================================
# TAB 1: Model Metrics
# =========================================================

with tab1:
    st.subheader("📊 Model Performance Comparison")
    
    # Create two columns for Prophet and SARIMAX
    col1, col2 = st.columns(2)
    
    # Prophet Section - Show Future Forecast Data
    with col1:
        st.markdown("### 🟦 Prophet")
        # Display Prophet performance metrics (from calculate_prophet_metrics_from_forecast)
        if 'prophet_metrics_dict' in locals() or 'prophet_metrics_dict' in globals():
            st.markdown("**Performance:**")
            metric_cols = st.columns(3)
            
            with metric_cols[0]:
                st.metric("MAE", f"{prophet_metrics_dict['MAE']:.2f}")
            with metric_cols[1]:
                st.metric("RMSE", f"{prophet_metrics_dict['RMSE']:.2f}")
            with metric_cols[2]:
                st.metric("MAPE", f"{prophet_metrics_dict['MAPE']:.2f}%")
        
        st.markdown("**Future Forecast Preview**")
        
        if not prophet_future.empty:
            # Show first 20 rows of future forecast
            display_df = prophet_future
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
        
    
    # SARIMAX Section - Show Metrics
    with col2:
        st.markdown("### 🟥 SARIMAX")
        
        if not metrics_df.empty:
            # Normalize column names
            metrics_df.columns = metrics_df.columns.str.upper()
            
            # Calculate and display average metrics BEFORE the table
            st.markdown("**Average Metrics:**")
            metric_cols = st.columns(3)
            
            avg_mae = metrics_df["MAE"].mean()
            avg_rmse = metrics_df["RMSE"].mean()
            avg_mape = metrics_df["MAPE"].mean()
            
            with metric_cols[0]:
                st.metric("MAE", f"{avg_mae:.2f}")
            with metric_cols[1]:
                st.metric("RMSE", f"{avg_rmse:.2f}")
            with metric_cols[2]:
                st.metric("MAPE", f"{avg_mape:.2f}%")
            
            st.markdown("**Model Metrics**")
            
            # Display SARIMAX metrics table
            st.dataframe(
                metrics_df.style.format(precision=2).background_gradient(
                    subset=["MAE", "RMSE", "MAPE"],
                    cmap="YlOrRd"
                ),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("⚠️ SARIMAX metrics data not loaded")

# =========================================================
# TAB 2: Rolling Forecast
# =========================================================

with tab2:
    st.subheader("📉 Divorce Forecast Comparison: SARIMAX vs Prophet")
    
    fig_all = go.Figure()
    
    # =========================
    # Actual
    # =========================
    if "Divorce" in df.columns and "ds" in df.columns:
        fig_all.add_trace(go.Scatter(
            x=df["ds"],
            y=df["Divorce"],
            name="Actual",
            line=dict(color=COLORS["actual"], width=2)
        ))
    
    # =========================
    # SARIMAX Rolling
    # =========================
    if not arima_roll.empty:
        fig_all.add_trace(go.Scatter(
            x=arima_roll["ds"],
            y=arima_roll["forecast"],
            name="SARIMAX Rolling",
            line=dict(color=COLORS["sarimax"], dash="dash")
        ))
    
    # =========================
    # SARIMAX Future
    # =========================
    if not arima_future.empty:
        fig_all.add_trace(go.Scatter(
            x=arima_future["ds"],
            y=arima_future["yhat"],
            name="SARIMAX Future",
            line=dict(color=COLORS["sarimax"], dash="dash", width=2)
        ))
    
    # =========================
    # Prophet Future
    # =========================
    if not prophet_future.empty:
        fig_all.add_trace(go.Scatter(
            x=prophet_future["ds"],
            y=prophet_future["yhat"],
            name="Prophet Future",
            line=dict(color=COLORS["prophet"], dash="dash", width=2)
        ))
    
    # =========================
    # Layout
    # =========================
    fig_all.update_layout(
        title={
            'text': "Divorce Forecast Comparison: SARIMAX vs Prophet",
            'font': {'size': 22, 'color': '#2C3E50', 'family': 'Arial Black'}
        },
        xaxis_title="Date",
        yaxis_title="Number of Divorces",
        hovermode="x unified",
        template="plotly_white",
        plot_bgcolor='rgba(240, 242, 245, 0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif", size=12, color="#2C3E50"),
        xaxis={'gridcolor': '#E1E8ED'},
        yaxis={'gridcolor': '#E1E8ED'},
        height=700
    )
    
    st.plotly_chart(fig_all, use_container_width=True)
    
    # Show forecast data table
    if show_data_tables:
        with st.expander("📋 View Forecast Data"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**SARIMAX Rolling Forecast**")
                if not arima_roll.empty:
                    st.dataframe(arima_roll.tail(20), use_container_width=True)
                else:
                    st.warning("⚠️ SARIMAX rolling forecast data not loaded")
            with col2:
                st.markdown("**Future Forecasts**")
                if not arima_future.empty:
                    st.dataframe(arima_future.head(20), use_container_width=True)
                if not prophet_future.empty:
                    st.dataframe(prophet_future.head(20), use_container_width=True)

# =========================================================
# TAB 3: Future Forecast
# =========================================================

with tab3:
    st.subheader("🔮 Future Forecast Predictions")
    
    # User input for forecast horizon
    max_forecast_months = max(len(prophet_future), len(arima_future)) if not prophet_future.empty or not arima_future.empty else 60
    
    forecast_months = st.slider(
        "Select number of months to display",
        min_value=12,
        max_value=max_forecast_months,
        value=min(24, max_forecast_months),
        step=12,
        help="Adjust the forecast horizon"
    )
    
    # Prepare actual data from divorce_all_model.csv (using 'Divorce' column)
    if "Divorce" in df.columns and "ds" in df.columns:
        actual_data = df[["ds", "Divorce"]].copy()
    else:
        actual_data = pd.DataFrame()
    
    # Create sub-tabs for each model
    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Prophet", "SARIMAX", "Combined View"])
    
    # Prophet Sub-tab
    with sub_tab1:
        if "Prophet" in models_to_show and not prophet_future.empty:
            fig_prophet = go.Figure()
            
            # Add actual values
            if not actual_data.empty:
                fig_prophet.add_trace(go.Scatter(
                    x=actual_data["ds"],
                    y=actual_data["Divorce"],
                    name="Actual (Historical)",
                    line=dict(color=COLORS["actual"], width=2)
                ))
            
            # Add Prophet forecast
            prophet_subset = prophet_future.head(forecast_months)
            fig_prophet.add_trace(go.Scatter(
                x=prophet_subset["ds"],
                y=prophet_subset["yhat"],
                name="Future Forecast",
                line=dict(color=COLORS["prophet"], dash="dash", width=2)
            ))
            
            # Add confidence intervals
            if show_confidence_intervals and "yhat_upper" in prophet_subset.columns and "yhat_lower" in prophet_subset.columns:
                fig_prophet.add_trace(go.Scatter(
                    x=prophet_subset["ds"],
                    y=prophet_subset["yhat_upper"],
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig_prophet.add_trace(go.Scatter(
                    x=prophet_subset["ds"],
                    y=prophet_subset["yhat_lower"],
                    fill="tonexty",
                    fillcolor="rgba(99, 110, 250, 0.15)",
                    line=dict(width=0),
                    name="Confidence Interval"
                ))
            
            fig_prophet.update_layout(
                title={
                    'text': f"Future Forecast of Divorce Cases (Prophet – {forecast_months} months)",
                    'font': {'size': 20, 'color': '#2C3E50', 'family': 'Arial Black'}
                },
                xaxis_title="Date",
                yaxis_title="Number of Divorces",
                hovermode="x unified",
                template="plotly_white",
                plot_bgcolor='rgba(240, 242, 245, 0.8)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Arial, sans-serif", size=12, color="#2C3E50"),
                xaxis={'gridcolor': '#E1E8ED'},
                yaxis={'gridcolor': '#E1E8ED'},
                height=600
            )
            
            st.plotly_chart(fig_prophet, use_container_width=True)
            
            # Show data table
            if show_data_tables:
                st.dataframe(prophet_subset, use_container_width=True)
        else:
            st.info("Prophet forecast data not available")
    
    # SARIMAX Sub-tab
    with sub_tab2:
        if "SARIMAX" in models_to_show and not arima_future.empty:
            fig_arima = go.Figure()
            
            # Add actual values
            if not actual_data.empty:
                fig_arima.add_trace(go.Scatter(
                    x=actual_data["ds"],
                    y=actual_data["Divorce"],
                    name="Actual (Historical)",
                    line=dict(color=COLORS["actual"], width=3)
                ))
            
            # Add SARIMAX forecast
            arima_subset = arima_future.head(forecast_months)
            fig_arima.add_trace(go.Scatter(
                x=arima_subset["ds"],
                y=arima_subset["yhat"],
                name="SARIMAX Forecast",
                line=dict(color=COLORS["sarimax"], width=2, dash="dash")
            ))
            
            # Add bounds if available
            if show_confidence_intervals and "cap" in arima_subset.columns and "floor" in arima_subset.columns:
                fig_arima.add_trace(go.Scatter(
                    x=arima_subset["ds"],
                    y=arima_subset["cap"],
                    name="Upper Bound (Cap)",
                    line=dict(color=COLORS["sarimax"], width=1, dash="dash"),
                    opacity=0.3
                ))
                fig_arima.add_trace(go.Scatter(
                    x=arima_subset["ds"],
                    y=arima_subset["floor"],
                    name="Lower Bound (Floor)",
                    line=dict(color=COLORS["sarimax"], width=1, dash="dash"),
                    fill='tonexty',
                    opacity=0.2
                ))
            
            fig_arima.update_layout(
                title={
                    'text': f"SARIMAX Future Forecast ({forecast_months} months)",
                    'font': {'size': 20, 'color': '#2C3E50', 'family': 'Arial Black'}
                },
                xaxis_title="Date",
                yaxis_title="Predicted Divorce Count",
                hovermode="x unified",
                template="plotly_white",
                plot_bgcolor='rgba(240, 242, 245, 0.8)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Arial, sans-serif", size=12, color="#2C3E50"),
                xaxis={'gridcolor': '#E1E8ED'},
                yaxis={'gridcolor': '#E1E8ED'},
                height=600
            )
            
            st.plotly_chart(fig_arima, use_container_width=True)
            
            # Show data table
            if show_data_tables:
                st.dataframe(arima_subset, use_container_width=True)
        else:
            st.info("SARIMAX forecast data not available")
    
    # Combined View Sub-tab
    with sub_tab3:
        st.markdown("**Combined Model Comparison**")
        
        fig_combined = go.Figure()
        
        # =========================
        # Actual
        # =========================
        if not actual_data.empty:
            fig_combined.add_trace(go.Scatter(
                x=actual_data["ds"],
                y=actual_data["Divorce"],
                name="Actual",
                line=dict(color=COLORS["actual"], width=2)
            ))
        
        # # =========================
        # # SARIMAX Rolling
        # # =========================
        # if not arima_roll.empty:
        #     fig_combined.add_trace(go.Scatter(
        #         x=arima_roll["ds"],
        #         y=arima_roll["forecast"],
        #         name="SARIMAX Rolling",
        #         line=dict(color=COLORS["sarimax"], dash="dot")
        #     ))
        
        # =========================
        # SARIMAX Future
        # =========================
        if "SARIMAX" in models_to_show and not arima_future.empty:
            arima_subset = arima_future.head(forecast_months)
            fig_combined.add_trace(go.Scatter(
                x=arima_subset["ds"],
                y=arima_subset["yhat"],
                name="SARIMAX Future",
                line=dict(color=COLORS["sarimax"], dash="dash", width=2)
            ))
        
        # =========================
        # Prophet Future
        # =========================
        if "Prophet" in models_to_show and not prophet_future.empty:
            prophet_subset = prophet_future.head(forecast_months)
            fig_combined.add_trace(go.Scatter(
                x=prophet_subset["ds"],
                y=prophet_subset["yhat"],
                name="Prophet Future",
                line=dict(color=COLORS["prophet"], dash="dash", width=2)
            ))
        
        # =========================
        # Layout
        # =========================
        fig_combined.update_layout(
            title={
                'text': "Divorce Forecast Comparison: SARIMAX vs Prophet",
                'font': {'size': 20, 'color': '#2C3E50', 'family': 'Arial Black'}
            },
            xaxis_title="Date",
            yaxis_title="Number of Divorces",
            hovermode="x unified",
            template="plotly_white",
            plot_bgcolor='rgba(240, 242, 245, 0.8)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif", size=12, color="#2C3E50"),
            xaxis={'gridcolor': '#E1E8ED'},
            yaxis={'gridcolor': '#E1E8ED'},
            height=700
        )
        
        st.plotly_chart(fig_combined, use_container_width=True)

# # =========================================================
# # TAB 4: Prophet Scenario Test
# # =========================================================

# with tab4:
#     st.subheader("🧪 Prophet Scenario Testing")
#     st.markdown("""
#     This tool allows you to test the Prophet model with scenario simulation based on historical monthly patterns.
#     The simulation generates synthetic future data using monthly statistics (mean, std, min, max) from a scenario period.
#     """)
    
#     # User Controls
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         sim_years = st.slider(
#             "Number of Simulation Years",
#             min_value=1,
#             max_value=10,
#             value=5,
#             help="How many years to simulate into the future"
#         )
    
#     with col2:
#         use_best_params = st.checkbox(
#             "Use Hyperparameter Tuning",
#             value=True,
#             help="Automatically find optimal Prophet parameters"
#         )
    
#     with col3:
#         show_components = st.checkbox(
#             "Show Monthly Statistics",
#             value=False,
#             help="Display monthly statistics table used for simulation"
#         )
    
#     # Load Scenario Data
#     try:
#         df_scenario = pd.read_csv(DATA_FILES["scenario"])
#         df_scenario['ds'] = pd.to_datetime(df_scenario['ds'])
        
#         # Calculate monthly statistics
#         monthly_stats = get_monthly_stats(df_scenario.copy())
        
#         if show_components:
#             st.markdown("**📊 Monthly Statistics (from Scenario Data)**")
#             st.dataframe(monthly_stats.style.background_gradient(cmap="Blues"), use_container_width=True)
        
#         # Generate Scenario Button
#         if st.button("🎲 Generate Scenario Forecast", type="primary"):
#             with st.spinner("Generating scenario and training model..."):
                
#                 # Get last date from historical data
#                 last_date = df['ds'].max()
                
#                 # Generate random scenario
#                 df_simulated = generate_random_scenario(
#                     monthly_stats=monthly_stats,
#                     start_date=last_date,
#                     sim_years=sim_years
#                 )
                
#                 # Data Augmentation: Combine real + simulated
#                 df_augmented = pd.concat([
#                     df_scenario[['ds', 'y']],
#                     df_simulated
#                 ], ignore_index=True)
                
#                 # Set Global Cap/Floor for Logistic Growth
#                 global_cap = df_augmented['y'].max() * 1.2
#                 global_floor = 0
                
#                 df_augmented['cap'] = global_cap
#                 df_augmented['floor'] = global_floor
                
#                 # Perform hyperparameter tuning if enabled
#                 if use_best_params:
#                     st.info("🔍 Performing hyperparameter tuning on scenario data...")
#                     best_params = tune_prophet_hyperparameters(df_augmented, global_cap, global_floor)
#                 else:
#                     # Use default parameters when tuning is disabled
#                     best_params = {
#                         'changepoint_prior_scale': 0.05,
#                         'seasonality_prior_scale': 10.0,
#                         'yearly_seasonality': True,
#                         'weekly_seasonality': False,
#                         'daily_seasonality': False
#                     }
#                     st.info("ℹ️ Using default parameters (tuning disabled)")
                
#                 # Display the parameters being used
#                 with st.expander("📋 View Model Parameters"):
#                     st.json(best_params)
                
#                 # Train Prophet Model with tuned/default parameters
#                 m_scenario = Prophet(
#                     growth="logistic",
#                     **best_params
#                 )
                
#                 m_scenario.fit(df_augmented)
                
#                 # Make Future Predictions
#                 future = m_scenario.make_future_dataframe(periods=60, freq='MS')
#                 future['cap'] = global_cap
#                 future['floor'] = global_floor
                
#                 forecast_scenario = m_scenario.predict(future)
                
#                 # Visualization
#                 st.success("✅ Scenario forecast generated successfully!")
                
#                 st.markdown("### 📈 Scenario Forecast Visualization")
                
#                 fig_scenario = go.Figure()
                
#                 # Add actual historical data
#                 fig_scenario.add_trace(go.Scatter(
#                     x=df_scenario['ds'],
#                     y=df_scenario['y'],
#                     name="Actual History (Scenario Period)",
#                     mode='lines',
#                     line=dict(color=COLORS["actual"], width=2)
#                 ))
                
#                 # Add simulated data line
#                 fig_scenario.add_trace(go.Scatter(
#                     x=df_simulated['ds'],
#                     y=df_simulated['y'],
#                     name="Simulated Data",
#                     mode='lines+markers',
#                     line=dict(color=COLORS["simulated"], width=2),
#                     marker=dict(color=COLORS["simulated"], size=5, opacity=0.7)
#                 ))
                
#                 # Add forecast line
#                 fig_scenario.add_trace(go.Scatter(
#                     x=forecast_scenario['ds'],
#                     y=forecast_scenario['yhat'],
#                     name="Prophet Scenario Forecast",
#                     line=dict(color=COLORS["prophet"], width=2, dash='dash')
#                 ))
                
#                 # Add confidence intervals
#                 if show_confidence_intervals:
#                     fig_scenario.add_trace(go.Scatter(
#                         x=forecast_scenario['ds'],
#                         y=forecast_scenario['yhat_upper'],
#                         name="Upper Bound",
#                         line=dict(color=COLORS["prophet"], width=1, dash='dot'),
#                         opacity=0.3
#                     ))
                    
#                     fig_scenario.add_trace(go.Scatter(
#                         x=forecast_scenario['ds'],
#                         y=forecast_scenario['yhat_lower'],
#                         name="Lower Bound",
#                         line=dict(color=COLORS["prophet"], width=1, dash='dot'),
#                         fill='tonexty',
#                         opacity=0.2
#                     ))
                
#                 fig_scenario.update_layout(
#                     title={
#                         'text': f"Prophet Scenario Test ({sim_years} Years Simulation)",
#                         'font': {'size': 22, 'color': '#2C3E50', 'family': 'Arial Black'}
#                     },
#                     xaxis_title="Date",
#                     yaxis_title="Divorce Count",
#                     hovermode="x unified",
#                     template="plotly_white",
#                     plot_bgcolor='rgba(240, 242, 245, 0.8)',
#                     paper_bgcolor='rgba(0,0,0,0)',
#                     font=dict(family="Arial, sans-serif", size=12, color="#2C3E50"),
#                     xaxis={'gridcolor': '#E1E8ED'},
#                     yaxis={'gridcolor': '#E1E8ED'},
#                     height=650
#                 )
                
#                 st.plotly_chart(fig_scenario, use_container_width=True)
                
#                 # Show statistics
#                 col1, col2, col3 = st.columns(3)
                
#                 with col1:
#                     st.metric(
#                         "Simulated Data Points",
#                         f"{len(df_simulated)}"
#                     )
                
#                 with col2:
#                     avg_simulated = df_simulated['y'].mean()
#                     st.metric(
#                         "Avg Simulated Value",
#                         f"{avg_simulated:,.0f}"
#                     )
                
#                 with col3:
#                     final_forecast = forecast_scenario['yhat'].iloc[-1]
#                     st.metric(
#                         "Final Forecast Value",
#                         f"{final_forecast:,.0f}"
#                     )
                
#                 # Show data tables
#                 if show_data_tables:
#                     with st.expander("📋 View Scenario Data Details"):
#                         tab_a, tab_b, tab_c = st.tabs([
#                             "Simulated Data",
#                             "Augmented Dataset",
#                             "Forecast Results"
#                         ])
                        
#                         with tab_a:
#                             st.markdown("**Randomly Generated Simulation Data**")
#                             st.dataframe(df_simulated, use_container_width=True)
                        
#                         with tab_b:
#                             st.markdown("**Combined Real + Simulated Data**")
#                             st.dataframe(df_augmented.tail(100), use_container_width=True)
                        
#                         with tab_c:
#                             st.markdown("**Prophet Forecast Output**")
#                             display_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']
#                             st.dataframe(
#                                 forecast_scenario[display_cols].tail(60),
#                                 use_container_width=True
#                             )
        
#         else:
#             st.info("👆 Click the button above to generate a scenario forecast")
            
#             # Show preview of scenario data
#             st.markdown("### 📊 Scenario Data Preview")
#             st.markdown(f"**Data Range:** {df_scenario['ds'].min().date()} to {df_scenario['ds'].max().date()}")
#             st.markdown(f"**Total Records:** {len(df_scenario)}")
            
#             # Preview chart
#             fig_preview = go.Figure()
#             fig_preview.add_trace(go.Scatter(
#                 x=df_scenario['ds'],
#                 y=df_scenario['y'],
#                 name="Scenario Data",
#                 line=dict(color=COLORS["actual"], width=2),
#                 fill='tozeroy',
#                 fillcolor='rgba(0,0,0,0.1)'
#             ))
            
#             fig_preview.update_layout(
#                 title={
#                     'text': "Scenario Data (Used for Monthly Statistics)",
#                     'font': {'size': 18, 'color': '#2C3E50', 'family': 'Arial Black'}
#                 },
#                 xaxis_title="Date",
#                 yaxis_title="Divorce Count",
#                 template="plotly_white",
#                 plot_bgcolor='rgba(240, 242, 245, 0.8)',
#                 paper_bgcolor='rgba(0,0,0,0)',
#                 font=dict(family="Arial, sans-serif", size=12, color="#2C3E50"),
#                 xaxis={'gridcolor': '#E1E8ED'},
#                 yaxis={'gridcolor': '#E1E8ED'},
#                 height=400
#             )
            
#             st.plotly_chart(fig_preview, use_container_width=True)
    
#     except FileNotFoundError:
#         st.error(f"""
#         ❌ **Scenario data file not found!**
        
#         Please make sure `{DATA_FILES['scenario']}` exists in the same directory.
#         This file should contain historical data with columns: `ds` (date) and `y` (value).
#         """)
#     except Exception as e:
#         st.error(f"❌ An error occurred: {str(e)}")

# =========================================================
# Footer
# =========================================================

st.divider()
st.caption(f"🔮 Forecast Models: Prophet • SARIMAX")
st.caption("📊 Enhanced with improved visualizations, error handling, and user experience")
st.caption("💡 Data Source: https://stat.bora.dopa.go.th/stat/statnew/statMenu/newStat/home.php")