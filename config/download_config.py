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

# Quick settings for different scenarios
QUICK_SETTINGS = {
    'conservative': {
        'requests_per_minute': 15,
        'min_delay_between_requests': 4.0,
        'max_files_per_session': 30,
        'session_break_duration': 600,  # 10 minutes
    },
    'aggressive': {
        'requests_per_minute': 30,
        'min_delay_between_requests': 2.0,
        'max_files_per_session': 100,
        'session_break_duration': 180,  # 3 minutes
    },
    'balanced': {
        'requests_per_minute': 20,
        'min_delay_between_requests': 3.0,
        'max_files_per_session': 50,
        'session_break_duration': 300,  # 5 minutes
    }
}

def apply_quick_setting(setting_name: str):
    """Apply a quick setting to the configuration"""
    if setting_name not in QUICK_SETTINGS:
        print(f"Unknown setting: {setting_name}")
        print(f"Available settings: {list(QUICK_SETTINGS.keys())}")
        return False
    
    setting = QUICK_SETTINGS[setting_name]
    
    # Apply to rate limit config
    for key, value in setting.items():
        if key in RATE_LIMIT_CONFIG:
            RATE_LIMIT_CONFIG[key] = value
        elif key in DOWNLOAD_CONFIG:
            DOWNLOAD_CONFIG[key] = value
    
    print(f"Applied {setting_name} settings:")
    print(f"  - Requests per minute: {RATE_LIMIT_CONFIG['requests_per_minute']}")
    print(f"  - Delay between requests: {RATE_LIMIT_CONFIG['min_delay_between_requests']}s")
    print(f"  - Max files per session: {DOWNLOAD_CONFIG['max_files_per_session']}")
    print(f"  - Session break duration: {DOWNLOAD_CONFIG['session_break_duration']}s")
    
    return True

def print_current_settings():
    """Print current configuration settings"""
    print("Current API Download Settings:")
    print("=" * 50)
    print(f"Rate Limiting:")
    print(f"  - Requests per minute: {RATE_LIMIT_CONFIG['requests_per_minute']}")
    print(f"  - Min delay: {RATE_LIMIT_CONFIG['min_delay_between_requests']}s")
    print(f"  - Max delay: {RATE_LIMIT_CONFIG['max_delay_between_requests']}s")
    print(f"  - Max retries: {RATE_LIMIT_CONFIG['max_retries']}")
    print(f"  - 404 retry delay: {RATE_LIMIT_CONFIG['404_retry_delay']}s")
    print(f"  - Max consecutive 404s: {RATE_LIMIT_CONFIG['max_consecutive_404s']}")
    print()
    print(f"Session Limits:")
    print(f"  - Max files per session: {DOWNLOAD_CONFIG['max_files_per_session']}")
    print(f"  - Session break duration: {DOWNLOAD_CONFIG['session_break_duration']}s")
    print(f"  - Session limits enabled: {DOWNLOAD_CONFIG['enable_session_limits']}")
    print()
    print(f"API Keys:")
    print(f"  - Max key switches: {API_KEY_CONFIG['max_key_switches']}")
    print(f"  - Key switch delay: {RATE_LIMIT_CONFIG['key_switch_delay']}s")

if __name__ == "__main__":
    print_current_settings()
    print("\nTo apply a quick setting, use:")
    print("  apply_quick_setting('conservative')  # For slow, safe downloads")
    print("  apply_quick_setting('balanced')      # For balanced downloads")
    print("  apply_quick_setting('aggressive')    # For fast downloads")
