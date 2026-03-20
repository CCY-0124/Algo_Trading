"""
download_config.py

Configuration file for Glassnode API download limits and settings.
Modify these values to adjust the download behavior.
"""

# Rate limiting configuration - Enhanced for better API handling
RATE_LIMIT_CONFIG = {
    'requests_per_minute': 10,  # Conservative limit to avoid rate limits
    'min_delay_between_requests': 6.0,  # 6 seconds delay between requests (60/10)
    'max_delay_between_requests': 300.0,  # 300 seconds (5 minutes) wait on rate limit
    'exponential_backoff_base': 2.0,
    'max_retries': 5,  # Increased retries
    'key_switch_delay': 15,  # Increased delay after key switch
    '404_retry_delay': 30,  # Special delay for 404 errors
    'max_consecutive_404s': 10,  # Max consecutive 404 errors before skipping
}

# API key configuration
API_KEY_CONFIG = {
    'services': ['glassnode'],
    'key_names': ['main', 'backup'],
    'current_key_index': 0,
    'key_switch_count': 0,
    'max_key_switches': 5,
}

# Download session configuration
DOWNLOAD_CONFIG = {
    'max_files_per_session': 50,  # Limit files per session to avoid API limits
    'session_break_duration': 300,  # 5 minutes break between sessions
    'enable_session_limits': True,  # Enable session limits
}

