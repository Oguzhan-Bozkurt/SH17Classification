import sys
import os
import time
import gc
import shutil
import argparse
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("❌ Eksik kütüphane: pip install ultralytics pandas")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================
os.environ['CUDNN_BENCHMARK'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.8,max_split_size_mb:128'

@dataclass
class Config:
    # Dinamik path'ler - çalıştırma sırasında ayarlanacak
    project_root: str = None
    images_dir: str = None
    
    # Dataset Sınıf Sıralaması
    class_names: List[str] = field(default_factory=lambda: [
        "person", "ear", "ear-mufs", "face", "face-guard", "face-mask",
        "foot", "tool", "glasses", "gloves", "helmet", "hands", "head",
        "medical-suit", "shoes", "safety-suit", "safety-vest"
    ])
    
    imgsz: int = 640
    batch_size: int = 16  # A5000
    workers: int = 4      # Hız için
    cache_images: bool = True # RAM Cache AÇIK

# ============================================================================
# UTILITIES
# ============================================================================
def prepare_dataset(cfg):
    root = Path(cfg.project_root)
    img_dir = Path(cfg.images_dir)
    new_files = {}
    print("\n🛠️ Dataset Hazırlanıyor...")
    print(f"📁 Proje Dizini: {root}")
    print(f"🖼️ Görsel Dizini: {img_dir}")
    
    for split in ['train', 'val', 'test']:
        txt = root / f"{split}.txt"
        abs_txt = root / f"{split}_abs.txt"
        if txt.exists():
            with open(txt, 'r') as f: lines = [l.strip() for l in f.readlines() if l.strip()]
            valid = []
            missing = 0
            for l in lines:
                p = Path(l)
                full = img_dir / p.name
                if full.exists(): 
                    valid.append(str(full.absolute()))
                elif p.exists(): 
                    valid.append(str(p.absolute()))
                else:
                    missing += 1
            with open(abs_txt, 'w') as f: f.write('\n'.join(valid))
            new_files[split] = str(abs_txt.absolute())
            print(f"   ✅ {split}: {len(valid)} dosya bulundu" + (f", {missing} eksik" if missing else ""))

    yaml_path = root / "sh17_single.yaml"
    with open(yaml_path, 'w') as f:
        for k, v in new_files.items(): f.write(f"{k}: {v}\n")
        f.write("\nnames:\n")
        for i, n in enumerate(cfg.class_names): f.write(f"  {i}: {n}\n")
    return yaml_path

def save_report(model_name, metrics, duration, total_epochs, save_dir):
    rows = []
    try:
        p = metrics.box.p
        r = metrics.box.r
        f1 = metrics.box.f1
        map50 = metrics.box.all_ap[:, 0]
        map95 = metrics.box.all_ap.mean(1)
        for i, name in metrics.names.items():
            rows.append({
                "Class": name, "Precision": round(p[i], 4), "Recall": round(r[i], 4),
                "F1-Score": round(f1[i], 4), "mAP@0.5": round(map50[i], 4), "mAP@0.5:0.95": round(map95[i], 4),
                "Duration_Min": ""
            })
    except: pass
    
    summary = {
        "Class": "TOTAL_SUMMARY",
        "Precision": round(metrics.box.mp, 4), "Recall": round(metrics.box.mr, 4),
        "F1-Score": round(metrics.box.f1.mean(), 4), "mAP@0.5": round(metrics.box.map50, 4), 
        "mAP@0.5:0.95": round(metrics.box.map, 4), "Duration_Min": round(duration, 2)
    }
    df = pd.DataFrame(rows + [summary])
    
    out_dir = Path(save_dir) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_name}_{total_epochs}ep_report.csv"
    df.to_csv(out_path, index=False)
    print(f"📊 Rapor Kaydedildi: {out_path}")

def force_patch_workers(ckpt_path):
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if 'train_args' in ckpt and isinstance(ckpt['train_args'], dict):
            ckpt['train_args']['workers'] = 0
            torch.save(ckpt, ckpt_path)
            return True
    except: pass
    return False

def select_model_menu():
    models = [
        {"name": "yolov10l", "file": "yolov10l.pt"},
        {"name": "yolov10m", "file": "yolov10m.pt"},
        {"name": "yolov10b", "file": "yolov10b.pt"},
        {"name": "yolo11l", "file": "yolo11l.pt"},
        {"name": "yolo11m", "file": "yolo11m.pt"},
        {"name": "yolo12l", "file": "yolo12l.pt"}, 
        {"name": "yolo12m", "file": "yolo12m.pt"},
    ]
    print("\n--------------------------------")
    print("MODEL SEÇİMİ")
    print("--------------------------------")
    for i, m in enumerate(models, 1):
        status = "✅ İndirilmiş" if os.path.exists(m['file']) else "⬇️ İndirilecek"
        print(f" {i}. {m['name']:<10} ({status})")
    
    while True:
        try:
            sel = int(input("\nModel Numarası Girin: ").strip())
            if 1 <= sel <= len(models):
                return models[sel-1]['name']
            print("❌ Geçersiz numara.")
        except ValueError:
            print("❌ Lütfen sayı girin.")

# ============================================================================
# MAIN TRAINING LOGIC
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='SH17 YOLO Eğitim Aracı')
    parser.add_argument('--project-root', type=str, 
                        default=os.environ.get('SH17_PROJECT_ROOT', '.'),
                        help='Proje ana dizini (train.txt, val.txt burada)')
    parser.add_argument('--images-dir', type=str,
                        default=os.environ.get('SH17_IMAGES_DIR', None),
                        help='Görsellerin bulunduğu dizin (varsayılan: project_root/data/images)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Config oluştur ve path'leri ayarla
    cfg = Config()
    cfg.project_root = os.path.abspath(args.project_root)
    cfg.images_dir = args.images_dir or os.path.join(cfg.project_root, 'data', 'images')
    
    # Path'leri doğrula
    if not os.path.isdir(cfg.images_dir):
        print(f"❌ HATA: Görsel dizini bulunamadı: {cfg.images_dir}")
        sys.exit(1)
    
    yaml_path = prepare_dataset(cfg)
    
    result_dir = Path("SH17_Results_Single")
    ckpt_dir = result_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*50)
    print("TEK MODEL EĞİTİM ARACI (V27)")
    print("="*50)
    
    # 1. Menüden Seçim
    model_name = select_model_menu()
    pt_file = f"{model_name}.pt"
    
    # 2. Mod Seçimi
    print("\n[R]esume (Devam Et) | [O]verwrite (Sıfırla)")
    mode = input("Seçim [R/o]: ").strip().upper()
    action = "OVERWRITE" if mode == 'O' else "RESUME"
    
    # 3. Epoch Hedefi
    try:
        extra_epochs = int(input("\nKaç epoch eğitilecek (örn: 50): ").strip())
    except: extra_epochs = 50

    # Yolları belirle
    run_dir = Path("runs/detect") / model_name
    last_pt = run_dir / "weights" / "last.pt"
    
    resume_flag = False
    workers = cfg.workers # Varsayılan: 8 (Hızlı)
    
    # --- BURASI DÜZELTİLDİ ---
    if action == "OVERWRITE":
        if run_dir.exists():
            try: shutil.rmtree(run_dir)
            except: pass
            print("🧹 Eski eğitim klasörü silindi.")
        
        current_weights = pt_file
        current_epoch = 0  # <--- EKLENEN SATIR: HATA BURADAYDI
        
    else:
        # Resume durumu
        if last_pt.exists():
            print(f"🔄 Kaldığı yer tespit edildi: {last_pt}")
            current_weights = str(last_pt)
            resume_flag = True
            
            try:
                ckpt = torch.load(last_pt, map_location='cpu')
                current_epoch = ckpt.get('epoch', -1) + 1
                print(f"ℹ️  Şu anki Epoch: {current_epoch}")
                
                workers = 0 
                force_patch_workers(last_pt)
            except: 
                current_epoch = 0
        else:
            print("⚠️ Kayıtlı model bulunamadı, sıfırdan başlanıyor...")
            current_weights = pt_file
            current_epoch = 0
    # --------------------------

    # Hedef Epoch Hesapla
    total_target_epochs = current_epoch + extra_epochs if resume_flag else extra_epochs
    
    print(f"\n🚀 MODEL: {model_name.upper()}")
    print(f"🎯 HEDEF: {current_epoch} -> {total_target_epochs} Epoch")
    print(f"⚡ AYARLAR: Cache={cfg.cache_images}, Workers={workers}")
    
    input("👉 Başlamak için ENTER...")
    
    if os.name == 'nt':
        torch.multiprocessing.set_sharing_strategy('file_system')

    gc.collect()
    torch.cuda.empty_cache()
    
    start_time = time.time()
    
    try:
        print(f"📥 Model yükleniyor: {current_weights}...")
        try:
            model = YOLO(current_weights)
        except Exception as e:
            print(f"\n❌ HATA: Model dosyası '{current_weights}' bulunamadı veya indirilemedi.")
            print(f"🛑 Teknik Detay: {e}")
            sys.exit(1)

        # EĞİTİM
        if resume_flag:
            print("🔄 Fine-Tuning Modu (Süreç uzatılıyor)...")
            model.train(
                data=str(yaml_path),
                epochs=total_target_epochs,
                imgsz=cfg.imgsz,
                batch=cfg.batch_size,
                workers=workers,
                device=0,
                project="runs/detect",
                name=model_name,
                exist_ok=True,
                resume=False, 
                cache=cfg.cache_images,
                patience=50
            )
        else:
            print("🆕 Sıfırdan Eğitim Başlıyor...")
            model.train(
                data=str(yaml_path),
                epochs=total_target_epochs,
                imgsz=cfg.imgsz,
                batch=cfg.batch_size,
                workers=workers,
                device=0,
                project="runs/detect",
                name=model_name,
                exist_ok=True,
                cache=cfg.cache_images,
                patience=50
            )
            
        # SONUÇLARI KAYDET
        final_last_pt = run_dir / "weights" / "last.pt"
        
        if final_last_pt.exists():
            print("📊 Rapor ve Yedek Oluşturuluyor...")
            
            del model
            gc.collect()
            torch.cuda.empty_cache()
            
            val_model = YOLO(str(final_last_pt))
            metrics = val_model.val(data=str(yaml_path), device=0, batch=cfg.batch_size, workers=0, verbose=False)
            
            duration = (time.time() - start_time) / 60.0
            save_report(model_name, metrics, duration, total_target_epochs, str(result_dir))
            
            dest_pt = ckpt_dir / f"{model_name}_{total_target_epochs}ep.pt"
            shutil.copy(final_last_pt, dest_pt)
            print(f"💾 Model Kaydedildi: {dest_pt}")

    except Exception as e:
        print(f"\n❌ BEKLENMEYEN HATA: {e}")

if __name__ == "__main__":
    main()