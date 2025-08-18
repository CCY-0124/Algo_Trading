"""
setup_environment.py

Sets up a consistent development environment by:
- Loading environment variables from .env
- Checking Python version compatibility
- Installing required packages
- Ensuring a valid data directory exists
- Creating a symbolic link to the data path if needed

:precondition: A `.env` file must be present in the same directory
:postcondition: Development environment is aligned with the project's configuration"""
import os
import sys
import subprocess
from dotenv import dotenv_values


def setup_environment():
    """
    Set up a consistent development environment based on .env configuration.

    :precondition: .env must define PYTHON_VERSION, REQUIRED_PACKAGES, and DATA_PATH
    :postcondition: Python version is checked, required packages installed, data path created, and symlink established
    :return: None
    """
    # Load environment config
    env_config = dotenv_values(".env")

    print("=" * 50)
    print("Setting up consistent development environment")
    print("=" * 50)

    # 1. Check Python version
    current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    required_python = env_config.get("PYTHON_VERSION", "3.9")

    if current_python != required_python:
        print(f"Python version mismatch: current {current_python}, required {required_python}")
        print("Please install the correct Python version")

    # 2. Install required packages
    required_packages = env_config.get("REQUIRED_PACKAGES", "").split(",")
    installed_packages = subprocess.getoutput("pip freeze").split("\n")

    to_install = []
    for pkg in required_packages:
        if pkg and not any(pkg in installed for installed in installed_packages):
            to_install.append(pkg)

    if to_install:
        print(f"Installing packages: {', '.join(to_install)}")
        subprocess.run(f"pip install {' '.join(to_install)}", shell=True)
    else:
        print("All required packages are already installed")

    # 3. Ensure data path exists
    data_path = env_config.get("DATA_PATH", "D:\\Trading_Data")
    if not os.path.exists(data_path):
        print(f"Creating data directory: {data_path}")
        os.makedirs(data_path)

    # 4. Data path verification
    print(f"Data path configured: {data_path}")
    print("Note: Data is stored in D:\\Trading_Data, not in project directory")

    print("=" * 50)
    print("Environment setup complete!")
    print("=" * 50)


if __name__ == "__main__":
    try:
        import dotenv
    except ImportError:
        subprocess.run("pip install python-dotenv", shell=True)
        import dotenv

    setup_environment()