"""
System Test Script for Dual-Model Backend
Tests all endpoints and verifies the Two-Lane Highway system is working
"""

import requests
import json
import time
from pathlib import Path
from PIL import Image
import io

# Configuration
API_BASE_URL = "http://127.0.0.1:8001"
TIMEOUT = 10

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def print_section(text):
    """Print formatted section"""
    print(f"\n📌 {text}")
    print("-"*60)

def test_health():
    """Test health endpoint"""
    print_section("Testing Health Endpoint")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()
        print(f"✅ Status: {response.status_code}")
        print(f"✅ Response: {json.dumps(data, indent=2)}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print_section("Testing Model Info Endpoint")
    try:
        response = requests.get(f"{API_BASE_URL}/model-info", timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()
        print(f"✅ Status: {response.status_code}")
        print(f"✅ System: {data.get('system')}")
        print(f"✅ Lane 1 Specialist: {data['lane_1_specialist']['name']}")
        print(f"   - Loaded: {data['lane_1_specialist']['loaded']}")
        print(f"   - Classes: {data['lane_1_specialist']['num_classes']}")
        print(f"✅ Lane 2 Generalist: {data['lane_2_generalist']['name']}")
        print(f"   - Loaded: {data['lane_2_generalist']['loaded']}")
        print(f"   - Classes: {data['lane_2_generalist']['num_classes']}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def create_test_image():
    """Create a test image file"""
    print_section("Creating Test Image")
    try:
        # Create a simple RGB image
        img = Image.new('RGB', (512, 512), color=(200, 100, 50))
        
        # Save to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        print(f"✅ Created test image: 512x512 RGB")
        return img_bytes
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_dual_prediction():
    """Test dual model prediction"""
    print_section("Testing Dual-Model Prediction Endpoint")
    
    # Create test image
    img_bytes = create_test_image()
    if not img_bytes:
        return False
    
    try:
        files = {'file': ('test_image.jpg', img_bytes, 'image/jpeg')}
        response = requests.post(
            f"{API_BASE_URL}/predict-dual",
            files=files,
            timeout=TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Status: {response.status_code}")
        print(f"✅ Response Status: {data.get('status')}")
        print(f"✅ System Mode: {data.get('mode')}")
        
        # Lane 1 Results
        if data.get('lane_1_specialist'):
            lane1 = data['lane_1_specialist']
            print(f"\n🚗 LANE 1 (Specialist - DR Detection):")
            print(f"   Disease: {lane1.get('disease')}")
            print(f"   Diagnosis: {lane1.get('diagnosis')}")
            print(f"   Confidence: {lane1.get('confidence'):.4f}")
            print(f"   Severity: {lane1.get('severity')}")
            print(f"   Recommended Action: {lane1.get('recommended_action')}")
            if lane1.get('probabilities'):
                print(f"   Probabilities:")
                for class_name, prob in lane1['probabilities'].items():
                    print(f"     - {class_name}: {prob:.4f}")
        
        # Lane 2 Results
        if data.get('lane_2_generalist'):
            lane2 = data['lane_2_generalist']
            print(f"\n🚗 LANE 2 (Generalist - Glaucoma/Cataract Detection):")
            print(f"   Disease: {lane2.get('disease')}")
            print(f"   Diagnosis: {lane2.get('diagnosis')}")
            print(f"   Confidence: {lane2.get('confidence'):.4f}")
            print(f"   Severity: {lane2.get('severity')}")
            print(f"   Recommended Action: {lane2.get('recommended_action')}")
            if lane2.get('probabilities'):
                print(f"   Probabilities:")
                for class_name, prob in lane2['probabilities'].items():
                    print(f"     - {class_name}: {prob:.4f}")
        
        # Combined Risk
        if data.get('combined_risk_level'):
            print(f"\n⚠️ COMBINED RISK LEVEL: {data['combined_risk_level']}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    print_header("DUAL-MODEL SYSTEM TEST SUITE")
    
    print("\n🚗🚗 Two-Lane Highway Backend Verification")
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Timeout: {TIMEOUT}s")
    
    # Wait for server to be ready
    print("\n⏳ Waiting for backend to be ready...")
    max_retries = 5
    for attempt in range(max_retries):
        try:
            requests.get(f"{API_BASE_URL}/health", timeout=2)
            print("✅ Backend is ready!\n")
            break
        except:
            if attempt < max_retries - 1:
                print(f"  Attempt {attempt+1}/{max_retries}... retrying in 2 seconds...")
                time.sleep(2)
            else:
                print("❌ Backend not responding. Make sure it's running!")
                return
    
    # Run tests
    results = {
        "Health Check": test_health(),
        "Model Info": test_model_info(),
        "Dual Prediction": test_dual_prediction(),
    }
    
    # Summary
    print_header("TEST SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready to use!")
        print("\n🌐 Frontend Setup:")
        print("   1. Open new terminal in frontend/ directory")
        print("   2. Run: npm install (if needed)")
        print("   3. Run: npm start")
        print("   4. Open http://localhost:3000")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above.")

if __name__ == "__main__":
    main()
