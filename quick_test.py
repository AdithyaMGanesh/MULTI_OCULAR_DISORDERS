#!/usr/bin/env python
"""Quick test of the dual-model backend"""

import time
import subprocess
import requests

def test_endpoints():
    """Test backend endpoints"""
    API_URL = "http://localhost:8001"
    
    print("\n" + "="*60)
    print("  DUAL-MODEL SYSTEM - QUICK TEST")
    print("="*60)
    
    # Wait for server
    print("\n⏳ Checking if backend is ready...")
    for i in range(10):
        try:
            resp = requests.get(f"{API_URL}/health", timeout=2)
            print("✅ Backend is online!\n")
            break
        except:
            if i < 9:
                print(f"  Waiting... ({i+1}/10)")
                time.sleep(1)
            else:
                print("❌ Backend not responding!")
                return False
    
    # Test 1: Health
    try:
        resp = requests.get(f"{API_URL}/health")
        print(f"✅ Health Check: {resp.json()['status']}")
        print(f"   - Specialist Model: {'✅ Loaded' if resp.json()['specialist_model_loaded'] else '⚠️ Not Loaded'}")
        print(f"   - Generalist Model: {'✅ Loaded' if resp.json()['generalist_model_loaded'] else '⚠️ Not Loaded'}")
        print(f"   - Mode: {resp.json()['system_mode']}")
    except Exception as e:
        print(f"❌ Health Check Failed: {e}")
        return False
    
    # Test 2: Model Info
    try:
        resp = requests.get(f"{API_URL}/model-info")
        data = resp.json()
        print(f"\n✅ Model Info Retrieved:")
        print(f"   System: {data['system']}")
        print(f"   Lane 1: {data['lane_1_specialist']['name']}")
        print(f"           {data['lane_1_specialist']['num_classes']} classes")
        print(f"   Lane 2: {data['lane_2_generalist']['name']}")
        print(f"           {data['lane_2_generalist']['num_classes']} classes")
    except Exception as e:
        print(f"❌ Model Info Failed: {e}")
        return False
    
    # Test 3: Dual Prediction
    try:
        from PIL import Image
        import io
        
        # Create a test image
        img = Image.new('RGB', (512, 512), color=(150, 100, 50))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        files = {'file': ('test.jpg', img_bytes, 'image/jpeg')}
        resp = requests.post(f"{API_URL}/predict-dual", files=files, timeout=30)
        data = resp.json()
        
        print(f"\n✅ Dual Prediction Results:")
        print(f"   Status: {data['status']}")
        print(f"   Mode: {data['mode']}")
        
        if 'lane_1_specialist' in data:
            L1 = data['lane_1_specialist']
            print(f"\n   🚗 Lane 1 (Specialist - DR):")
            print(f"      Diagnosis: {L1.get('diagnosis')}")
            print(f"      Confidence: {L1.get('confidence'):.2%}")
            print(f"      Severity: {L1.get('severity')}")
        
        if 'lane_2_generalist' in data:
            L2 = data['lane_2_generalist']
            print(f"\n   🚗 Lane 2 (Generalist):")
            print(f"      Diagnosis: {L2.get('diagnosis')}")
            print(f"      Confidence: {L2.get('confidence'):.2%}")
            print(f"      Severity: {L2.get('severity')}")
        
        if 'combined_risk_level' in data:
            print(f"\n   ⚠️ Combined Risk: {data['combined_risk_level']}")
        
    except Exception as e:
        print(f"❌ Dual Prediction Failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("  ✅ ALL TESTS PASSED!")
    print("="*60)
    print("\n📌 Next Steps:")
    print("   1. The backend is running on http://localhost:8001")
    print("   2. API Docs available at: http://localhost:8001/docs")
    print("   3. To start frontend: cd frontend && npm install && npm start")
    print("   4. Open http://localhost:3000 in your browser")
    print("\n")
    
    return True

if __name__ == "__main__":
    try:
        test_endpoints()
    except KeyboardInterrupt:
        print("\n\n🛑 Test interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
