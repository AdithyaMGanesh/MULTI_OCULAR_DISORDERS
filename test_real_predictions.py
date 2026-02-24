#!/usr/bin/env python3
"""
Quick test to verify backend is making REAL predictions, not DEMO mode.
"""
import requests
import os

# Create a simple test image
from PIL import Image
import numpy as np

# Generate a random test image (512x512 RGB)
test_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8), 'RGB')
test_img_path = '/tmp/test_image.png'
test_img.save(test_img_path)

# Send to backend
url = 'http://localhost:8003/predict-dual'
files = {'file': open(test_img_path, 'rb')}

try:
    response = requests.post(url, files=files)
    data = response.json()
    
    print("\n" + "="*80)
    print("BACKEND PREDICTION TEST")
    print("="*80)
    
    if 'mode' in data:
        mode = data['mode'].upper()
        if mode == 'PRODUCTION':
            print(f"✅ MODE: PRODUCTION (Real predictions)")
        else:
            print(f"❌ MODE: {mode} (Demo/Test mode)")
    
    print(f"\nStatus: {data.get('status', 'N/A')}")
    
    # Check Lane 1
    if data.get('lane_1_specialist'):
        lane1 = data['lane_1_specialist']
        print(f"\n🚗 LANE 1 (DR Detection):")
        print(f"   Diagnosis: {lane1.get('diagnosis')}")
        print(f"   Confidence: {lane1.get('confidence'):.4f}")
        print(f"   Severity: {lane1.get('severity')}")
        if lane1.get('gradcam'):
            print(f"   GradCAM: ✅ Generated")
    
    # Check Lane 2
    if data.get('lane_2_generalist'):
        lane2 = data['lane_2_generalist']
        print(f"\n🚗 LANE 2 (Glaucoma/Cataract):")
        print(f"   Diagnosis: {lane2.get('diagnosis')}")
        print(f"   Confidence: {lane2.get('confidence'):.4f}")
        print(f"   Severity: {lane2.get('severity')}")
    
    print(f"\nCombined Risk: {data.get('combined_risk_level')}")
    print("\n" + "="*80)
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure backend is running on port 8003")
