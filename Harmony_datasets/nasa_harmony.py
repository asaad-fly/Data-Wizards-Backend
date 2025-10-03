import asyncio
from typing import Optional
import numpy as np
import xarray as xr
from harmony import BBox, Client, Collection, Request
from harmony.config import Environment
from functools import lru_cache


# ==================== Harmony Client Setup ====================
@lru_cache()
def get_harmony_client():
    """Initialize Harmony client with credentials"""
    return Client(env=Environment.PROD)

# ==================== Data Fetching ====================
async def fetch_satellite_data(
    lat: float, 
    lon: float, 
    collection: str,
    buffer: float = 0.5
) -> Optional[xr.Dataset]:
    """
    Fetch satellite data from NASA Harmony for a specific location
    No time dimension - gets latest available data
    """
    client = get_harmony_client()
    
    # Create bounding box around point
    bbox = BBox(lon - buffer, lat - buffer, lon + buffer, lat + buffer)
    
    request = Request(
        collection=Collection(id=collection),
        spatial=bbox,
    )
    
    try:
        job_id = client.submit(request)
        
        # Wait for job completion
        for _ in range(30):  # 30 second timeout
            await asyncio.sleep(1)
            status = client.status(job_id)
            if status.get('status') == 'successful':
                break
        
        # Download results
        results = client.result_urls(job_id)
        if results:
            ds = xr.open_dataset(results[0])
            return ds
    except Exception as e:
        print(f"Harmony API error: {str(e)}")
        return None
    
    return None

def extract_pollutant_at_location(
    dataset: xr.Dataset, 
    lat: float, 
    lon: float,
    variable_name: str
) -> float:
    """
    Extract pollutant concentration at specific lat/lon
    Dataset has only lat/lon dimensions (no time)
    """
    try:
        # Select nearest point to location
        point_data = dataset.sel(lat=lat, lon=lon, method='nearest')
        
        if variable_name in point_data.data_vars:
            value = float(point_data[variable_name].values)
            return value
        else:
            print(f"Variable {variable_name} not found in dataset")
            return np.nan
    except Exception as e:
        print(f"Error extracting data: {str(e)}")
        return np.nan
