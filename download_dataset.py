import os
import urllib.request
import zipfile
import shutil

CONDITIONS = ["TA", "TS", "KA", "KS"]
FRAMES = ["01", "05", "10", "15", "20"]

DATA_DIR = "data"

def download_albanian():
    # Albanian ID
    base = ("https://www.kaggle.com/api/v1/datasets/download/"
            "kontheeboonmeeprakob/midv500?file_name=midv500%2F01_alb_id")
    img_dir = os.path.join(DATA_DIR, "alb_id", "images")
    os.makedirs(img_dir, exist_ok=True)

    for cond in CONDITIONS:
        for fr in FRAMES:
            fname = f"{cond}01_{fr}.tif"
            dst = os.path.join(img_dir, fname)
            if os.path.exists(dst):
                continue   # already downloaded, skip
            url = f"{base}%2Fimages%2F{cond}%2F{fname}"
            tmp = dst + ".zip"
            try:
                urllib.request.urlretrieve(url, tmp)
                # each file is zipped; extracting the real .tif from inside
                with zipfile.ZipFile(tmp) as zf:
                    with zf.open(zf.namelist()[0]) as src, open(dst, "wb") as out:
                        shutil.copyfileobj(src, out)
                os.remove(tmp)
            except Exception as e:
                print(f"  FAIL {fname}: {e}")

def download_romanian():
    # Romanian driving licence
    url = "ftp://smartengines.com/midv-500/dataset/38_rou_drvlic.zip"
    zip_path = os.path.join(DATA_DIR, "38_rou_drvlic.zip")

    # skip if already downloaded
    if os.path.exists(os.path.join(DATA_DIR, "38_rou_drvlic", "images")):
        print("  deja descarcat, sar peste")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as e:
        print(f"  FAIL: {e}")
        return

    # extract only the clean-background conditions, skip everything else
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if any(f"/images/{c}/" in member for c in CONDITIONS):
                zf.extract(member, DATA_DIR)
    os.remove(zip_path)

if __name__ == "__main__":
    download_albanian()
    download_romanian()