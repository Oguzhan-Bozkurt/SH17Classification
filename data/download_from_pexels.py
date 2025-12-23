import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Ayarlar
MAX_WORKERS = 128  # Aynı anda kaç indirme yapılsın (internet hızına göre ayarla)
TIMEOUT = 30      # İndirme timeout (saniye)

# Progress için global sayaç
progress_lock = Lock()
completed_count = 0
total_count = 0

def download_image(url):
    """Tek bir görsel indir"""
    global completed_count
    
    url = url.strip()
    if not url:
        return None
    
    filename = url.split('/')[-1]
    filepath = f"images/{filename}"
    
    # Zaten varsa atla
    if os.path.exists(filepath):
        with progress_lock:
            completed_count += 1
        return f"⏭️ {filename} (zaten var)"
    
    try:
        r = requests.get(url, allow_redirects=True, timeout=TIMEOUT)
        r.raise_for_status()
        with open(filepath, 'wb') as f:
            f.write(r.content)
        
        with progress_lock:
            completed_count += 1
            pct = 100 * completed_count / total_count
            print(f"\r⬇️ İndiriliyor: {completed_count}/{total_count} ({pct:.1f}%)", end="", flush=True)
        
        return f"✅ {filename}"
    except Exception as e:
        with progress_lock:
            completed_count += 1
        return f"❌ {filename}: {e}"

def main():
    global total_count
    
    with open("list_of_all_urls.csv", "r") as file:
        urls = [u.strip() for u in file.readlines() if u.strip()]
    
    total_count = len(urls)
    os.makedirs("images", exist_ok=True)
    
    print(f"🚀 {total_count} görsel {MAX_WORKERS} paralel bağlantı ile indiriliyor...\n")
    
    failed = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_image, url): url for url in urls}
        
        for future in as_completed(futures):
            result = future.result()
            if result and result.startswith("❌"):
                failed.append(result)
    
    print(f"\n\n{'='*50}")
    print(f"✅ Tamamlandı: {completed_count}/{total_count}")
    
    if failed:
        print(f"❌ Başarısız: {len(failed)}")
        for f in failed[:10]:  # İlk 10 hatayı göster
            print(f"   {f}")
        if len(failed) > 10:
            print(f"   ... ve {len(failed)-10} hata daha")

if __name__ == '__main__':
    main()
