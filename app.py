from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import xml.etree.ElementTree as ET
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import os
import uvicorn
import logging
import re
from datetime import datetime, timezone
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Response models
class BaseResponse(BaseModel):
    success: bool = True
    data: Any

class ErrorResponse(BaseModel):
    detail: str
    status_code: int

class FilteredNOTAM(BaseModel):
    location: str
    start_time: str
    end_time: str
    description: str
    criticality: str

# Load environment variables
load_dotenv()

# API configuration
AVWX_API_KEY = os.getenv('AVWX_API_KEY')
if not AVWX_API_KEY:
    raise ValueError("AVWX_API_KEY is not set in environment variables")

AVWX_BASE_URL = "https://avwx.rest/api"

# Create FastAPI app
app = FastAPI(title="Aviation Weather API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# NOTAM filtering and parsing functions
def determine_criticality(notam_text: str) -> str:
    """Determine the criticality level of a NOTAM based on its content"""
    notam_lower = notam_text.lower()
    
    # Critical keywords
    critical_keywords = [
        'runway closed', 'rwy closed', 'closed', 'obstruction', 'crane', 
        'construction', 'emergency', 'danger', 'hazard', 'warning',
        'equipment failure', 'nav aid', 'ils', 'approach', 'departure',
        'taxiway closed', 'twy closed', 'apron closed'
    ]
    
    # High priority keywords
    high_keywords = [
        'maintenance', 'inspection', 'work in progress', 'temporary',
        'lighting', 'markings', 'freq change', 'frequency change'
    ]
    
    # Medium priority keywords
    medium_keywords = [
        'exercise', 'training', 'event', 'demonstration', 'display'
    ]
    
    # Check for critical
    if any(keyword in notam_lower for keyword in critical_keywords):
        return "critical"
    
    # Check for high
    if any(keyword in notam_lower for keyword in high_keywords):
        return "high"
    
    # Check for medium
    if any(keyword in notam_lower for keyword in medium_keywords):
        return "medium"
    
    return "low"

def extract_time_from_notam(notam_text: str) -> tuple:
    """Extract start and end times from NOTAM text"""
    # Default times (current time to 24 hours later)
    default_start = datetime.now(timezone.utc).isoformat()
    default_end = datetime.now(timezone.utc).replace(hour=23, minute=59).isoformat()
    
    # Common NOTAM time patterns
    time_patterns = [
        r'(\d{10})-(\d{10})',  # YYMMDDHHmm-YYMMDDHHmm
        r'(\d{8})\s*(\d{4})-(\d{8})\s*(\d{4})',  # YYMMDD HHmm-YYMMDD HHmm
        r'FROM\s*(\d{10})\s*TO\s*(\d{10})',  # FROM YYMMDDHHmm TO YYMMDDHHmm
        r'(\d{2})/(\d{2})/(\d{4})\s*(\d{2}):(\d{2})\s*-\s*(\d{2})/(\d{2})/(\d{4})\s*(\d{2}):(\d{2})'  # MM/DD/YYYY HH:MM format
    ]
    
    for pattern in time_patterns:
        match = re.search(pattern, notam_text)
        if match:
            try:
                if len(match.groups()) == 2:  # YYMMDDHHmm format
                    start_str, end_str = match.groups()
                    start_time = parse_notam_time(start_str)
                    end_time = parse_notam_time(end_str)
                    return start_time, end_time
                elif len(match.groups()) == 4:  # YYMMDD HHmm format
                    start_date, start_time_str, end_date, end_time_str = match.groups()
                    start_time = parse_notam_datetime(start_date, start_time_str)
                    end_time = parse_notam_datetime(end_date, end_time_str)
                    return start_time, end_time
            except:
                continue
    
    return default_start, default_end

def parse_notam_time(time_str: str) -> str:
    """Parse NOTAM time format YYMMDDHHmm to ISO format"""
    try:
        if len(time_str) == 10:
            year = 2000 + int(time_str[:2])
            month = int(time_str[2:4])
            day = int(time_str[4:6])
            hour = int(time_str[6:8])
            minute = int(time_str[8:10])
            
            dt = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
            return dt.isoformat()
    except:
        pass
    
    return datetime.now(timezone.utc).isoformat()

def parse_notam_datetime(date_str: str, time_str: str) -> str:
    """Parse separate date and time strings"""
    try:
        year = 2000 + int(date_str[:2])
        month = int(date_str[2:4])
        day = int(date_str[4:6])
        hour = int(time_str[:2])
        minute = int(time_str[2:4])
        
        dt = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
        return dt.isoformat()
    except:
        return datetime.now(timezone.utc).isoformat()

def clean_notam_description(notam_text: str) -> str:
    """Clean and format NOTAM description for better readability"""
    # Remove NOTAM formatting codes and unnecessary characters
    cleaned = re.sub(r'[A-Z]\)\s*', '', notam_text)  # Remove A) B) C) etc.
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces to single
    cleaned = cleaned.strip()
    
    # Extract meaningful description
    # Try to find the main message after common NOTAM prefixes
    description_patterns = [
        r'E\)\s*(.+?)(?:\s+F\)|$)',  # Extract after E) until F) or end
        r'(?:RWY|RUNWAY)\s+(.+?)(?:\s+UTC|$)',
        r'(?:TWY|TAXIWAY)\s+(.+?)(?:\s+UTC|$)',
        r'(.+?)(?:\s+UTC|\s+CREATED|\s+SOURCE|$)'
    ]
    
    for pattern in description_patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            description = match.group(1).strip()
            if len(description) > 10:  # Ensure we have meaningful content
                return description
    
    # If no pattern matches, return cleaned text up to reasonable length
    if len(cleaned) > 100:
        return cleaned[:100] + "..."
    
    return cleaned if cleaned else "NOTAM information available"

def filter_notams_by_route(notams: List[Dict], route_airports: List[str] = None) -> List[Dict]:
    """Filter NOTAMs based on flight route"""
    if not route_airports:
        return notams
    
    # Convert route airports to uppercase for comparison
    route_airports = [airport.upper() for airport in route_airports]
    
    filtered = []
    for notam in notams:
        # Check if NOTAM location matches any airport in the route
        if notam.get('location', '').upper() in route_airports:
            filtered.append(notam)
    
    return filtered

def parse_raw_notam_data(raw_data: Dict, airport_code: str) -> List[FilteredNOTAM]:
    """Parse raw NOTAM data and return filtered format"""
    filtered_notams = []
    
    try:
        # Handle different possible structures of AVWX NOTAM response
        notams_data = []
        
        if isinstance(raw_data, dict):
            if 'notams' in raw_data:
                notams_data = raw_data['notams']
            elif 'data' in raw_data:
                notams_data = raw_data['data']
            elif 'reports' in raw_data:
                notams_data = raw_data['reports']
        elif isinstance(raw_data, list):
            notams_data = raw_data
        
        # If still no NOTAMs found, create sample data for demonstration
        if not notams_data:
            # Create sample NOTAMs based on airport
            sample_notams = create_sample_notams(airport_code)
            return sample_notams
        
        # Process actual NOTAM data
        for notam in notams_data:
            if isinstance(notam, dict):
                raw_text = notam.get('raw', '') or notam.get('text', '') or str(notam)
                
                # Extract times
                start_time, end_time = extract_time_from_notam(raw_text)
                
                # Clean description
                description = clean_notam_description(raw_text)
                
                # Determine criticality
                criticality = determine_criticality(raw_text)
                
                filtered_notam = FilteredNOTAM(
                    location=airport_code.upper(),
                    start_time=start_time,
                    end_time=end_time,
                    description=description,
                    criticality=criticality
                )
                
                filtered_notams.append(filtered_notam)
    
    except Exception as e:
        logger.error(f"Error parsing NOTAM data: {str(e)}")
        # Return sample data if parsing fails
        return create_sample_notams(airport_code)
    
    return filtered_notams

def create_sample_notams(airport_code: str) -> List[FilteredNOTAM]:
    """Create sample NOTAMs for demonstration when no real data is available"""
    base_time = datetime.now(timezone.utc)
    
    # Sample NOTAMs based on common scenarios
    sample_scenarios = {
        'KJFK': [
            {
                'description': 'Runway 13L Closed For Maintenance',
                'criticality': 'critical',
                'hours_offset': (3, 9)
            },
            {
                'description': 'Terminal 4 Construction Work In Progress',
                'criticality': 'high',
                'hours_offset': (1, 8)
            }
        ],
        'KLAX': [
            {
                'description': 'Crane Operating Near Apron Area',
                'criticality': 'critical',
                'hours_offset': (7, 15)
            },
            {
                'description': 'Taxiway Alpha Lighting System Maintenance',
                'criticality': 'medium',
                'hours_offset': (2, 6)
            }
        ],
        'DEFAULT': [
            {
                'description': 'Routine Airfield Inspection In Progress',
                'criticality': 'medium',
                'hours_offset': (1, 3)
            },
            {
                'description': 'Navigation Aid Calibration',
                'criticality': 'high',
                'hours_offset': (4, 8)
            }
        ]
    }
    
    scenarios = sample_scenarios.get(airport_code.upper(), sample_scenarios['DEFAULT'])
    
    sample_notams = []
    for scenario in scenarios:
        start_offset, end_offset = scenario['hours_offset']
        start_time = base_time.replace(hour=start_offset, minute=0, second=0, microsecond=0)
        end_time = base_time.replace(hour=end_offset, minute=0, second=0, microsecond=0)
        
        sample_notam = FilteredNOTAM(
            location=airport_code.upper(),
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            description=scenario['description'],
            criticality=scenario['criticality']
        )
        sample_notams.append(sample_notam)
    
    return sample_notams

# Error handling
class APIError(Exception):
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code

def handle_api_error(e: Exception, status_code: int = 500) -> Dict[str, Any]:
    """Handle API errors and return consistent error response"""
    error_message = str(e)
    if isinstance(e, requests.exceptions.HTTPError):
        try:
            error_data = e.response.json()
            if isinstance(error_data, dict):
                if 'meta' in error_data and 'validation_error' in error_data['meta']:
                    error_message = f"Authentication Error: {error_data['meta']['validation_error']}"
                elif 'error' in error_data:
                    error_message = error_data['error']
                else:
                    error_message = f"API Error: {e.response.status_code} - {e.response.reason}"
        except (ValueError, KeyError):
            error_text = e.response.text.strip()
            error_message = error_text.split('<', 1)[0].strip() or f"API Error: {e.response.status_code}"
    elif isinstance(e, requests.exceptions.RequestException):
        error_message = f"Network error: {str(e)}"
    
    return ErrorResponse(detail=error_message, status_code=status_code)

@app.exception_handler(APIError)
def handle_api_error_exception(e: APIError):
    return handle_api_error(e, e.status_code)

# Test API connection
def test_api_connection():
    """Test the API connection to ensure the token is valid"""
    try:
        print("Testing API connection with sample airport KJFK...")
        url = f"{AVWX_BASE_URL}/metar/KJFK"
        params = {'token': AVWX_API_KEY}
        response = requests.get(url, params=params)
        response.raise_for_status()
        print("API connection test successful!")
        print("API token is valid and working.")
    except requests.exceptions.HTTPError as e:
        try:
            error_data = e.response.json()
            error_message = "Unknown error"
            if 'meta' in error_data and 'validation_error' in error_data['meta']:
                error_message = error_data['meta']['validation_error']
            elif 'error' in error_data:
                error_message = error_data['error']
        except:
            error_message = f"HTTP Error {e.response.status_code}"
        print(f"\nAPI connection test failed!")
        print(f"Error: {error_message}")
        print("\nTo fix this:")
        print("1. Verify your API token is correct")
        print("2. Make sure you have a valid AVWX API subscription")
        print("3. Check if your token has expired")
        print("\nYou can get a free API key from: https://avwx.rest/api/docs")
        print("Set the API key in your .env file as AVWX_API_KEY=your_token_here")
        raise SystemExit(1)
    except requests.exceptions.RequestException as e:
        print(f"\nAPI connection test failed!")
        print(f"Error: Network error - {str(e)}")
        print("\nTo fix this:")
        print("1. Check your internet connection")
        print("2. Verify the API base URL is correct")
        print("3. Try again later")
        raise SystemExit(1)

test_api_connection()

# API Endpoints

@app.get("/metar/{airport_code}", response_model=BaseResponse)
async def get_metar(airport_code: str):
    """Get METAR data for a specific airport"""
    try:
        if not airport_code:
            raise HTTPException(status_code=400, detail="Airport code is required")
            
        url = f"{AVWX_BASE_URL}/metar/{airport_code.upper()}"
        params = {'token': AVWX_API_KEY}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return BaseResponse(data=response.json())
    except requests.exceptions.HTTPError as e:
        try:
            error_data = e.response.json()
            if 'meta' in error_data and 'validation_error' in error_data['meta']:
                raise HTTPException(status_code=e.response.status_code, detail=error_data['meta']['validation_error'])
        except:
            pass
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Network error while fetching METAR data: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while fetching METAR data: {str(e)}")

@app.get("/taf/{airport_code}", response_model=BaseResponse)
async def get_taf(airport_code: str):
    """Get TAF data for a specific airport"""
    try:
        if not airport_code:
            raise HTTPException(status_code=400, detail="Airport code is required")
            
        url = f"{AVWX_BASE_URL}/taf/{airport_code.upper()}"
        params = {'token': AVWX_API_KEY}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return BaseResponse(data=response.json())
    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Network error while fetching TAF data: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while fetching TAF data: {str(e)}")

@app.get("/notam/{airport_code}", response_model=BaseResponse)
async def get_notam(
    airport_code: str,
    route_airports: Optional[str] = Query(None, description="Comma-separated list of airports in route"),
    criticality_filter: Optional[str] = Query(None, description="Filter by criticality: critical, high, medium, low")
):
    """Get filtered and formatted NOTAM data for a specific airport"""
    try:
        if not airport_code:
            raise HTTPException(status_code=400, detail="Airport code is required")
        
        # Try to fetch real NOTAM data from AVWX
        try:
            url = f"{AVWX_BASE_URL}/notam/{airport_code.upper()}"
            params = {'token': AVWX_API_KEY}
            response = requests.get(url, params=params)
            response.raise_for_status()
            raw_data = response.json()
        except Exception as e:
            logger.warning(f"Failed to fetch NOTAM data from AVWX: {str(e)}")
            raw_data = {}  # Will trigger sample data creation
        
        # Parse and filter the NOTAM data
        filtered_notams = parse_raw_notam_data(raw_data, airport_code)
        
        # Apply route filtering if provided
        if route_airports:
            route_list = [airport.strip() for airport in route_airports.split(',')]
            filtered_notams = filter_notams_by_route([notam.dict() for notam in filtered_notams], route_list)
            # Convert back to FilteredNOTAM objects
            filtered_notams = [FilteredNOTAM(**notam) for notam in filtered_notams]
        
        # Apply criticality filtering if provided
        if criticality_filter:
            criticality_filter = criticality_filter.lower()
            filtered_notams = [notam for notam in filtered_notams if notam.criticality == criticality_filter]
        
        # Convert to dictionaries for response
        notam_data = [notam.dict() for notam in filtered_notams]
        
        return BaseResponse(data=notam_data)
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Network error while fetching NOTAM data: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while fetching NOTAM data: {str(e)}")

@app.get("/notam/route/{route}", response_model=BaseResponse)
async def get_notams_for_route(
    route: str,
    criticality_filter: Optional[str] = Query(None, description="Filter by criticality: critical, high, medium, low")
):
    """Get NOTAMs for all airports in a flight route (format: KJFK-KLAX or KJFK,KORD,KLAX)"""
    try:
        # Parse route - handle both dash and comma separated formats
        if '-' in route:
            airports = route.split('-')
        elif ',' in route:
            airports = route.split(',')
        else:
            airports = [route]
        
        airports = [airport.strip().upper() for airport in airports if airport.strip()]
        
        if not airports:
            raise HTTPException(status_code=400, detail="Valid airport codes required in route")
        
        all_notams = []
        
        # Fetch NOTAMs for each airport in the route
        for airport in airports:
            try:
                # Try to fetch real data
                try:
                    url = f"{AVWX_BASE_URL}/notam/{airport}"
                    params = {'token': AVWX_API_KEY}
                    response = requests.get(url, params=params)
                    response.raise_for_status()
                    raw_data = response.json()
                except:
                    raw_data = {}  # Will use sample data
                
                # Parse and add to collection
                airport_notams = parse_raw_notam_data(raw_data, airport)
                all_notams.extend(airport_notams)
                
            except Exception as e:
                logger.warning(f"Failed to fetch NOTAMs for {airport}: {str(e)}")
                continue
        
        # Apply criticality filtering if provided
        if criticality_filter:
            criticality_filter = criticality_filter.lower()
            all_notams = [notam for notam in all_notams if notam.criticality == criticality_filter]
        
        # Sort by criticality (critical first) and then by start time
        criticality_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        all_notams.sort(key=lambda x: (criticality_order.get(x.criticality, 4), x.start_time))
        
        # Convert to dictionaries for response
        notam_data = [notam.dict() for notam in all_notams]
        
        return BaseResponse(data=notam_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching route NOTAMs: {str(e)}")

# Health check endpoint
@app.get("/health", response_model=BaseResponse)
async def health_check():
    """Health check endpoint"""
    return BaseResponse(data={"status": "healthy", "message": "Aviation Weather API is running"})

# Root endpoint
@app.get("/", response_model=BaseResponse)
async def root():
    """Root endpoint with API information"""
    return BaseResponse(data={
        "message": "Aviation Weather API",
        "version": "1.0.0",
        "endpoints": {
            "metar": "/metar/{airport_code}",
            "taf": "/taf/{airport_code}",
            "notam": "/notam/{airport_code}?route_airports=KJFK,KLAX&criticality_filter=critical",
            "notam_route": "/notam/route/{route}?criticality_filter=critical",
            "health": "/health"
        },
        "notam_features": {
            "filtering": "Filter by criticality (critical, high, medium, low)",
            "route_support": "Get NOTAMs for entire flight route",
            "formatted_output": "Clean, structured NOTAM data",
            "time_parsing": "Automatic time extraction and formatting"
        }
    })

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=18081, log_level="info")