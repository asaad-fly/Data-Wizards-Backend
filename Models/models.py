from pydantic import BaseModel
from typing import Optional
import numpy as np


# ==================== Models ====================
class Location(BaseModel):
    latitude: float
    longitude: float
    name: Optional[str] = None

class PollutantData(BaseModel):
    NO2_column: Optional[float] = None  # molecules/cm²
    NO2_ppb: Optional[float] = None
    NO2_aqi: Optional[int] = None
    O3_ppb: Optional[float] = None
    O3_aqi: Optional[int] = None
    HCHO_column: Optional[float] = None  # molecules/cm²
    HCHO_ppb: Optional[float] = None
    HCHO_aqi: Optional[int] = None

class AQIResponse(BaseModel):
    aqi: int
    category: str
    category_number: int  # 0-5
    pollutants: PollutantData
    timestamp: str
    location: Location
    health_recommendation: str
    dominant_pollutant: str

class WeatherData(BaseModel):
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    wind_speed: Optional[float] = None
    pressure: Optional[float] = None
    timestamp: str

# ==================== AQI Calculator (Your Implementation) ====================
class AQICalculator:
    """Calculate AQI based on EPA standards"""
    
    # EPA AQI Breakpoints for NO2 (ppb - 1-hour average)
    NO2_BREAKPOINTS = [
        (0, 53, 0, 50),          # Good
        (54, 100, 51, 100),      # Moderate
        (101, 360, 101, 150),    # Unhealthy for Sensitive
        (361, 649, 151, 200),    # Unhealthy
        (650, 1249, 201, 300),   # Very Unhealthy
        (1250, 2049, 301, 500),  # Hazardous
    ]
    
    # EPA AQI Breakpoints for O3 (ppb - 8-hour average)
    O3_BREAKPOINTS = [
        (0, 54, 0, 50),          # Good
        (55, 70, 51, 100),       # Moderate
        (71, 85, 101, 150),      # Unhealthy for Sensitive
        (86, 105, 151, 200),     # Unhealthy
        (106, 200, 201, 300),    # Very Unhealthy
    ]
    
    # HCHO approximate health-based thresholds
    HCHO_BREAKPOINTS = [
        (0, 10, 0, 50),          # Good
        (11, 30, 51, 100),       # Moderate
        (31, 50, 101, 150),      # Unhealthy for Sensitive
        (51, 80, 151, 200),      # Unhealthy
        (81, 120, 201, 300),     # Very Unhealthy
    ]
    
    @staticmethod
    def calculate_aqi(concentration, breakpoints):
        """
        Calculate AQI using EPA formula with official breakpoints
        Formula: AQI = [(I_high - I_low) / (C_high - C_low)] * (C - C_low) + I_low
        """
        # Convert to scalar if it's an array
        if isinstance(concentration, np.ndarray):
            concentration = float(concentration)
        
        if np.isnan(concentration) or concentration < 0:
            return np.nan
        
        # Find the appropriate breakpoint range
        for bp_low, bp_high, aqi_low, aqi_high in breakpoints:
            if bp_low <= concentration <= bp_high:
                aqi = ((aqi_high - aqi_low) / (bp_high - bp_low)) * (concentration - bp_low) + aqi_low
                return int(round(aqi))
        
        # If concentration exceeds all breakpoints, return highest category
        return breakpoints[-1][3]
    
    @staticmethod
    def no2_column_to_ppb(column_density):
        """Convert NO2 column density to approximate surface ppb"""
        # Simplified conversion for tropospheric NO2
        if np.isnan(column_density):
            return np.nan
        return column_density / 2e15  # Rough approximation
    
    @staticmethod
    def hcho_column_to_ppb(column_density):
        """Convert HCHO column density to approximate ppb"""
        if np.isnan(column_density):
            return np.nan
        return column_density / 1e16  # Rough approximation
    
    @classmethod
    def get_no2_aqi(cls, no2_column):
        """Calculate AQI for NO2"""
        ppb = cls.no2_column_to_ppb(no2_column)
        return cls.calculate_aqi(ppb, cls.NO2_BREAKPOINTS)
    
    @classmethod
    def get_o3_aqi(cls, o3_ppb):
        """Calculate AQI for O3"""
        return cls.calculate_aqi(o3_ppb, cls.O3_BREAKPOINTS)
    
    @classmethod
    def get_hcho_aqi(cls, hcho_column):
        """Calculate AQI for HCHO"""
        ppb = cls.hcho_column_to_ppb(hcho_column)
        return cls.calculate_aqi(ppb, cls.HCHO_BREAKPOINTS)
    
    @classmethod
    def get_combined_aqi(cls, no2_col, hcho_col, o3_ppb):
        """
        Get combined AQI - take the MAXIMUM (worst) pollutant AQI
        This follows EPA guidelines
        """
        aqis = {}
        
        no2_aqi = cls.get_no2_aqi(no2_col)
        if not np.isnan(no2_aqi):
            aqis['NO2'] = no2_aqi
        
        hcho_aqi = cls.get_hcho_aqi(hcho_col)
        if not np.isnan(hcho_aqi):
            aqis['HCHO'] = hcho_aqi
        
        o3_aqi = cls.get_o3_aqi(o3_ppb)
        if not np.isnan(o3_aqi):
            aqis['O3'] = o3_aqi
        
        if len(aqis) == 0:
            return np.nan, None
        
        # Find dominant pollutant (worst AQI)
        dominant = max(aqis, key=aqis.get)
        return max(aqis.values()), dominant
    
    @staticmethod
    def aqi_to_category(aqi):
        """Convert AQI value to category (0-5) and description"""
        if np.isnan(aqi):
            return np.nan, "Unknown", "No data available"
        
        if aqi <= 50:
            return 0, "Good", "Air quality is satisfactory, and air pollution poses little or no risk."
        elif aqi <= 100:
            return 1, "Moderate", "Air quality is acceptable. However, there may be a risk for some people."
        elif aqi <= 150:
            return 2, "Unhealthy for Sensitive Groups", "Members of sensitive groups may experience health effects."
        elif aqi <= 200:
            return 3, "Unhealthy", "Some members of the general public may experience health effects."
        elif aqi <= 300:
            return 4, "Very Unhealthy", "Health alert: The risk of health effects is increased for everyone."
        else:
            return 5, "Hazardous", "Health warning of emergency conditions: everyone is more likely to be affected."
