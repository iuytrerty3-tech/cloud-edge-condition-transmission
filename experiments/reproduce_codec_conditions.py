"""Reproduce per-branch condition payload / SSIM / encode-time statistics
on the 31 held-out evaluation images.

Usage:  python reproduce_codec_conditions.py [PROTOTYPE_ROOT]
        PROTOTYPE_ROOT defaults to the repository root (parent of this file),
        or set the CLOUD_EDGE_SD_ROOT environment variable.
"""
import io, json, os, sys, time
import numpy as np, cv2
from PIL import Image

_here = os.path.dirname(os.path.abspath(__file__))
ROOT = (sys.argv[1] if len(sys.argv) > 1 else
        os.environ.get("CLOUD_EDGE_SD_ROOT",
                       os.path.dirname(_here)))
ROOT = os.path.abspath(ROOT)
META = json.load(open(os.path.join(ROOT, "datasets", "starter_cultural_patterns", "paper_main_metadata.json"), encoding="utf-8"))
EVAL = set(json.load(open(os.path.join(ROOT, "experiments", "fullreal_eval_ids_v1.json"), encoding="utf-8")))

def to_local(p):
    # Metadata stores the original absolute path; remap it onto ROOT so the
    # script runs on any machine that has the dataset checkout.
    marker = "cloud_edge_sd_prototype"
    if marker in p:
        return os.path.join(ROOT, *p.split(marker, 1)[1].strip("/\\").split("/"))
    return p

def cond_of(path,size=512,lo=100,hi=200):
    img=Image.open(path).convert("RGB").resize((size,size),Image.Resampling.BILINEAR)
    rgb=np.asarray(img); gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    e=cv2.Canny(gray,lo,hi); return rgb, np.repeat(e[:,:,None],3,axis=2)
K=cv2.getGaussianKernel(11,1.5); WIN=K@K.T
def ssim3(a,b):  # HxWx3 uint8, cv2.filter2D handles 3 channels
    a=a.astype(np.float64); b=b.astype(np.float64); C1=(0.01*255)**2; C2=(0.03*255)**2
    ma=cv2.filter2D(a,-1,WIN); mb=cv2.filter2D(b,-1,WIN)
    ma2=ma*ma; mb2=mb*mb; mab=ma*mb
    sa=cv2.filter2D(a*a,-1,WIN)-ma2; sb=cv2.filter2D(b*b,-1,WIN)-mb2; sab=cv2.filter2D(a*b,-1,WIN)-mab
    m=((2*mab+C1)*(2*sab+C2))/((ma2+mb2+C1)*(sa+sb+C2)); return float(m.mean())
def pb(a,l): bf=io.BytesIO(); Image.fromarray(a).save(bf,format="PNG",compress_level=l); return bf.getvalue()
def jb(a,q): bf=io.BytesIO(); Image.fromarray(a).save(bf,format="JPEG",quality=q); return bf.getvalue()
ev=[m for m in META if m["id"] in EVAL]
B={}
def acc(n,p,s,e): B.setdefault(n,{"p":[],"s":[],"e":[]}); B[n]["p"].append(p); B[n]["s"].append(s); B[n]["e"].append(e)
t0=time.time()
for m in ev:
    rgb,cond=cond_of(to_local(m["image"]))
    for l in (1,3,6,9):
        t=time.perf_counter(); b=pb(cond,l); e=time.perf_counter()-t
        dec=np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
        acc(f"cond_png_l{l}_ds1", len(b)/1024, 1.0 if np.array_equal(dec,cond) else ssim3(cond,dec), e)
    for q in (75,95):
        t=time.perf_counter(); b=jb(cond,q); e=time.perf_counter()-t
        dec=np.asarray(Image.open(io.BytesIO(b)).convert("RGB")); acc(f"cond_jpeg_q{q}_ds1", len(b)/1024, ssim3(cond,dec), e)
    t=time.perf_counter(); b=pb(rgb,6); e=time.perf_counter()-t; acc("cloud_png", len(b)/1024, 1.0, e)
    t=time.perf_counter(); b=jb(rgb,75); e=time.perf_counter()-t
    dr=np.asarray(Image.open(io.BytesIO(b)).convert("RGB")); g=cv2.cvtColor(dr,cv2.COLOR_RGB2GRAY); c2=np.repeat(cv2.Canny(g,100,200)[:,:,None],3,2)
    acc("cloud_jpeg_q75", len(b)/1024, ssim3(cond,c2), e)
out={k:{"payload_kb":round(np.mean(v["p"]),3),"ssim":round(np.mean(v["s"]),5),"enc_ms":round(np.mean(v["e"])*1000,3)} for k,v in B.items()}
for k in sorted(out): print(f"{k:20} pay={out[k]['payload_kb']:8.2f}KB ssim={out[k]['ssim']:.5f} enc={out[k]['enc_ms']:.2f}ms")
print(f"elapsed {time.time()-t0:.1f}s")
json.dump(out, open(os.path.join(_here, "codec_repro_eval31.json"), "w"), indent=2)
