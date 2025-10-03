from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from typing import Optional
import datetime as dt
import numpy as np
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from Models.models import AQICalculator, AQIResponse, Location, PollutantData
from Harmony_datasets.nasa_harmony import fetch_satellite_data, extract_pollutant_at_location

app = FastAPI( 
    title="Data Wizards Backend API",
    description="Cloud-based Earth observation data for air quality prediction",
    version="1.0.0"
    )

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== API Endpoints ====================

@app.get("/")
async def root():
    return {
        "message": "AQI Prediction API for NASA Space Apps Challenge",
        "version": "1.0.0",
        "pollutants": ["NO2", "O3", "HCHO"],
        "endpoints": {
            "current_aqi": "/aqi/current?lat={latitude}&lon={longitude}",
            "health": "/health"
        }
    }

# Custom exception handler for request validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    logger.error(f"Request URL: {request.url}")
    logger.error(f"Query Params: {request.query_params}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Invalid request parameters",
            "errors": exc.errors(),
            "query_params": dict(request.query_params),
            "url": str(request.url)
        },
    )

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )

@app.get("/aqi/current", response_model=AQIResponse)
async def get_current_aqi(
    request: Request,
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    location_name: Optional[str] = Query(None, description="Location name")
):
    """
    Get current AQI for a location using NO2, O3, and HCHO data
    Data has only lat/lon dimensions (no time)
    """
    logger.info(f"Received request: {request.url}")
    logger.info(f"Query params: lat={lat}, lon={lon}, location_name={location_name}")
    
    # Validate coordinates
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid coordinates: lat must be between -90 and 90, lon between -180 and 180. Got lat={lat}, lon={lon}"
        )
    
    collections = {
        'NO2': 'C3685668972-LARC_CLOUD',  
        'O3': 'C2930764281-LARC_CLOUD',  
        'HCHO': 'C2930730944-LARC_CLOUD'
    }
    
    # Variable names in datasets
    variable_names = {
        'NO2': 'vertical_column_troposphere',
        'O3': 'o3_below_cloud',
        'HCHO': 'vertical_column'
    }
    
    pollutant_data = {}
    
    try:
        logger.info("Starting to fetch satellite data...")
        # Fetch NO2 data
        logger.info("Fetching NO2 data...")
        no2_ds = await fetch_satellite_data(lat, lon, collections['NO2'])
        if no2_ds:
            no2_column = extract_pollutant_at_location(no2_ds, lat, lon, variable_names['NO2'])
        else:
            # Mock data for demo if Harmony fails
            no2_column = np.random.uniform(1e15, 5e15)
        
        pollutant_data['NO2_column'] = no2_column
        pollutant_data['NO2_ppb'] = AQICalculator.no2_column_to_ppb(no2_column)
        pollutant_data['NO2_aqi'] = AQICalculator.get_no2_aqi(no2_column)
        
        # Fetch O3 data
        o3_ds = await fetch_satellite_data(lat, lon, collections['O3'])
        if o3_ds:
            o3_ppb = extract_pollutant_at_location(o3_ds, lat, lon, variable_names['O3'])
        else:
            # Mock data for demo
            o3_ppb = np.random.uniform(30, 70)
        
        pollutant_data['O3_ppb'] = o3_ppb
        pollutant_data['O3_aqi'] = AQICalculator.get_o3_aqi(o3_ppb)
        
        # Fetch HCHO data
        hcho_ds = await fetch_satellite_data(lat, lon, collections['HCHO'])
        if hcho_ds:
            hcho_column = extract_pollutant_at_location(hcho_ds, lat, lon, variable_names['HCHO'])
        else:
            # Mock data for demo
            hcho_column = np.random.uniform(1e15, 3e16)
        
        pollutant_data['HCHO_column'] = hcho_column
        pollutant_data['HCHO_ppb'] = AQICalculator.hcho_column_to_ppb(hcho_column)
        pollutant_data['HCHO_aqi'] = AQICalculator.get_hcho_aqi(hcho_column)
        
        # Log the collected data
        logger.info("Successfully collected all pollutant data")
        logger.info(f"NO2: {pollutant_data.get('NO2_ppb')} ppb (AQI: {pollutant_data.get('NO2_aqi')})")
        logger.info(f"O3: {pollutant_data.get('O3_ppb')} ppb (AQI: {pollutant_data.get('O3_aqi')})")
        logger.info(f"HCHO: {pollutant_data.get('HCHO_ppb')} ppb (AQI: {pollutant_data.get('HCHO_aqi')})")
        
        # Calculate combined AQI
        overall_aqi, dominant = AQICalculator.get_combined_aqi(
            no2_column, 
            hcho_column, 
            o3_ppb
        )
        
        if np.isnan(overall_aqi):
            raise HTTPException(status_code=500, detail="Unable to calculate AQI - no valid data")
        
        category_num, category_name, recommendation = AQICalculator.aqi_to_category(overall_aqi)
        
        return AQIResponse(
            aqi=int(overall_aqi),
            category=category_name,
            category_number=int(category_num),
            pollutants=PollutantData(**pollutant_data),
            timestamp=dt.datetime.utcnow().isoformat(),
            location=Location(latitude=lat, longitude=lon, name=location_name),
            health_recommendation=recommendation,
            dominant_pollutant=dominant if dominant else "Unknown"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail={
                "error": str(e),
                "type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
        )

@app.get("/aqi/grid")
async def get_aqi_grid(
    lat_min: float = Query(..., ge=-90, le=90),
    lat_max: float = Query(..., ge=-90, le=90),
    lon_min: float = Query(..., ge=-180, le=180),
    lon_max: float = Query(..., ge=-180, le=180),
    resolution: float = Query(0.1, gt=0, le=1, description="Grid resolution in degrees")
):
    """
    Get AQI data for a grid of points (for mapping)
    Useful for creating heatmaps
    """
    
    # Generate grid of points
    lats = np.arange(lat_min, lat_max, resolution)
    lons = np.arange(lon_min, lon_max, resolution)
    
    grid_data = []
    
    # Sample a subset of points to avoid timeout
    sample_points = min(100, len(lats) * len(lons))
    
    for _ in range(sample_points):
        lat = np.random.uniform(lat_min, lat_max)
        lon = np.random.uniform(lon_min, lon_max)
        
        # Mock AQI for grid (in production, fetch real data)
        aqi = np.random.randint(20, 150)
        category_num, category_name, _ = AQICalculator.aqi_to_category(aqi)
        
        grid_data.append({
            "lat": float(lat),
            "lon": float(lon),
            "aqi": int(aqi),
            "category": category_name
        })
    
    return {
        "bounds": {
            "lat_min": lat_min,
            "lat_max": lat_max,
            "lon_min": lon_min,
            "lon_max": lon_max
        },
        "resolution": resolution,
        "data": grid_data
    }

@app.get("/pollutants/{pollutant}")
async def get_single_pollutant(
    pollutant: str,
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180)
):
    """Get data for a single pollutant (NO2, O3, or HCHO)"""
    
    pollutant = pollutant.upper()
    
    if pollutant not in ['NO2', 'O3', 'HCHO']:
        raise HTTPException(status_code=400, detail="Pollutant must be NO2, O3, or HCHO")
    
    # Mock data (replace with real Harmony fetch)
    if pollutant == 'NO2':
        column = np.random.uniform(1e15, 5e15)
        ppb = AQICalculator.no2_column_to_ppb(column)
        aqi = AQICalculator.get_no2_aqi(column)
        return {
            "pollutant": "NO2",
            "column_density": float(column),
            "ppb": float(ppb),
            "aqi": int(aqi),
            "unit": "molecules/cm²"
        }
    elif pollutant == 'O3':
        ppb = np.random.uniform(30, 70)
        aqi = AQICalculator.get_o3_aqi(ppb)
        return {
            "pollutant": "O3",
            "ppb": float(ppb),
            "aqi": int(aqi),
            "unit": "ppb"
        }
    else:  # HCHO
        column = np.random.uniform(1e15, 3e16)
        ppb = AQICalculator.hcho_column_to_ppb(column)
        aqi = AQICalculator.get_hcho_aqi(column)
        return {
            "pollutant": "HCHO",
            "column_density": float(column),
            "ppb": float(ppb),
            "aqi": int(aqi),
            "unit": "molecules/cm²"
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": dt.datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/greet")
def greet():
    return "Hello from data-wizards-backend!"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
