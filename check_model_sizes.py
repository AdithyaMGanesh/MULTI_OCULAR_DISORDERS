import os
root = r'E:/DOWNLOADS/multi-ocular'
files = ['fusion_dr_model.keras','fusion_dr_model.h5','fusion_dr_model_final.keras']
for f in files:
    p = os.path.join(root,f)
    if os.path.exists(p):
        sz = os.path.getsize(p)
        print(f, sz, 'bytes', f"{sz/1024/1024:.2f} MB")
    else:
        print(f, 'MISSING')
