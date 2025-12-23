#!/usr/bin/env python3
"""
SH17 - YOLO Model İndirme Scripti
Tüm modelleri önceden indirir, eğitim sırasında bekleme olmaz.
"""

import os
import sys

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ ultralytics kütüphanesi bulunamadı!")
    print("   Yüklemek için: pip install ultralytics")
    sys.exit(1)

# İndirilecek modeller (train.py ile aynı liste)
MODELS = [
    "yolov10n.pt",
    "yolov10s.pt",
    "yolov10x.pt",
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11x.pt",
    "yolo12x.pt",
    "yolo12n.pt",
    "yolo12s.pt",
]

def download_all_models():
    print("=" * 50)
    print("🚀 YOLO MODEL İNDİRİCİ")
    print("=" * 50)
    print(f"📦 Toplam {len(MODELS)} model indirilecek\n")
    
    success = []
    failed = []
    
    for i, model_name in enumerate(MODELS, 1):
        print(f"[{i}/{len(MODELS)}] {model_name}...")
        
        if os.path.exists(model_name):
            print(f"   ✅ Zaten mevcut, atlanıyor.\n")
            success.append(model_name)
            continue
        
        try:
            # YOLO modeli yüklendiğinde otomatik indirilir
            model = YOLO(model_name)
            print(f"   ✅ Başarıyla indirildi!\n")
            success.append(model_name)
            del model  # Belleği temizle
        except Exception as e:
            print(f"   ❌ HATA: {e}\n")
            failed.append(model_name)
    
    # Özet
    print("=" * 50)
    print("📊 ÖZET")
    print("=" * 50)
    print(f"✅ Başarılı: {len(success)}/{len(MODELS)}")
    
    if success:
        print("   " + ", ".join(success))
    
    if failed:
        print(f"\n❌ Başarısız: {len(failed)}/{len(MODELS)}")
        print("   " + ", ".join(failed))
        print("\n⚠️ Bazı modeller indirilemedi. İnternet bağlantınızı kontrol edin.")
    else:
        print("\n🎉 Tüm modeller hazır! Eğitime başlayabilirsiniz.")

if __name__ == "__main__":
    download_all_models()

