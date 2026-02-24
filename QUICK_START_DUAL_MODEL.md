# 🚀 Quick Start Guide - Two-Lane Highway System

## 60-Second Setup

### Step 1: Start Backend (Specialist + Generalist Models)
```bash
# Windows CMD
START_BACKEND_DUAL.bat

# Windows PowerShell
.\START_BACKEND_DUAL.ps1

# macOS/Linux
python -m uvicorn src.api_dual_model:app --reload --host 0.0.0.0 --port 8001
```

Wait for message: `✅ READY: Dual-Model system is operational`

### Step 2: Start Frontend (React UI)
In a NEW terminal:
```bash
# Windows CMD
START_FRONTEND.bat

# Windows PowerShell
.\START_FRONTEND.ps1

# macOS/Linux
cd frontend && npm start
```

Wait for message: `Compiled successfully!` and browser opens

### Step 3: Use the System
1. Open http://localhost:3000
2. Upload a retinal image
3. Click "Analyze Image (Dual Model)"
4. See results from both lanes!

---

## What's Running?

| Component | Port | URL | Role |
|-----------|------|-----|------|
| Backend API | 8001 | http://localhost:8001 | Dual-model inference |
| Frontend UI | 3000 | http://localhost:3000 | React dashboard |
| API Docs | 8001 | http://localhost:8001/docs | Interactive API explorer |

---

## The Two Lanes Explained

### 🚗 Lane 1: DR Specialist
- Input: 224×224 image
- Disease: Diabetic Retinopathy
- Classes: No DR, Mild, Moderate, Severe, Proliferative
- Model: fusion_dr_model.h5

### 🚗 Lane 2: Glaucoma/Cataract Generalist
- Input: 128×128 image
- Disease: Glaucoma/Cataract
- Classes: Normal, Glaucoma, Cataract
- Model: custom_lightweight_model_v2.h5

---

## Demo Mode

**No model files?** System automatically runs in DEMO MODE:
- Still accepts image uploads
- Returns realistic random predictions
- Perfect for testing UI/UX

---

## Troubleshooting

### Backend errors?
```
Port 8001 already in use:
  → Kill process: netstat -ano | findstr :8001
  
Module not found:
  → pip install fastapi uvicorn tensorflow
  
CORS errors:
  → Check frontend URL is http://localhost:3000
```

### Frontend won't load?
```
Blank page or connection error:
  → Ensure backend is running first
  → Check http://localhost:8001/health
  → Clear browser cache (Ctrl+Shift+Delete)
```

### No model predictions?
```
Running in DEMO MODE (no .h5 files found):
  → Place models in root:
     - fusion_dr_model.h5
     - custom_lightweight_model_v2.h5
  → Restart backend
  → Check http://localhost:8001/model-info
```

---

## API Testing

### Health Check
```bash
curl http://localhost:8001/health
```

### Model Info
```bash
curl http://localhost:8001/model-info
```

### Predict (using curl)
```bash
curl -X POST http://localhost:8001/predict-dual \
  -F "file=@your_image.jpg"
```

---

## Next Steps

1. **Upload real retinal images** to see actual predictions
2. **Check http://localhost:8001/docs** for full API documentation
3. **Read README_DUAL_MODEL.md** for detailed information
4. **Explore the code** in `src/api_dual_model.py` and `frontend/src/AppDualModel.js`

---

## Key Features

✅ Parallel dual-model inference  
✅ Real-time combined risk assessment  
✅ Color-coded severity indicators  
✅ Probability distribution charts  
✅ Clinical recommendations  
✅ Responsive design  
✅ Demo mode (no models needed)  
✅ RESTful API with Swagger docs  

---

**Status:** 🟢 Ready to Run  
**System:** Two-Lane Highway v2.0  
**Models:** Optional (works in DEMO MODE without them)

Enjoy! 🎉
