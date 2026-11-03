import urllib.request
import time
import os
import sys

URL = "https://archive.ics.uci.edu/static/public/967/data.csv"
OUT_FILE = os.path.join("data", "phiusiil_dataset.csv")

def download_data():
    print(f"Connecting to {URL}...")
    req = urllib.request.Request(
        URL,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    )
    start_time = time.time()
    with urllib.request.urlopen(req, timeout=30) as response, open(OUT_FILE, 'wb') as out_f:
        downloaded = 0
        chunk_size = 1024 * 512 # 512 KB chunks
        last_print = time.time()
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)
            now = time.time()
            if now - last_print > 3:
                mb = downloaded / (1024 * 1024)
                speed = mb / (now - start_time)
                print(f"Downloaded {mb:.1f} MB ({speed:.2f} MB/s)...")
                last_print = now

    elapsed = time.time() - start_time
    total_mb = downloaded / (1024 * 1024)
    print(f"Download complete: {total_mb:.2f} MB in {elapsed:.1f}s ({total_mb/elapsed:.2f} MB/s)")

if __name__ == "__main__":
    download_data()
