"""
Download soccer_line model weights from GitHub releases.
Run this script after cloning the repository.
"""
import os
import urllib.request

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
BASE_URL = "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0"

WEIGHTS = {
    "MV_kp": f"{BASE_URL}/MV_kp",
    "MV_lines": f"{BASE_URL}/MV_lines",
}

def download_weights():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    for name, url in WEIGHTS.items():
        dest = os.path.join(WEIGHTS_DIR, name)
        if os.path.exists(dest):
            print(f"✓ {name} already exists")
            continue
        
        print(f"Downloading {name}...")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"✓ Downloaded {name}")
        except Exception as e:
            print(f"✗ Failed to download {name}: {e}")

if __name__ == "__main__":
    download_weights()
