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
    batch_size: int = 64  # A5000
    workers: int = 64     # Hız için
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
        # YOLOv10 Serisi
        {"name": "yolov10n", "file": "yolov10n.pt"},
        {"name": "yolov10s", "file": "yolov10s.pt"},
        {"name": "yolov10m", "file": "yolov10m.pt"},
        {"name": "yolov10b", "file": "yolov10b.pt"},
        {"name": "yolov10l", "file": "yolov10l.pt"},
        {"name": "yolov10x", "file": "yolov10x.pt"},
        # YOLO11 Serisi
        {"name": "yolo11n", "file": "yolo11n.pt"},
        {"name": "yolo11s", "file": "yolo11s.pt"},
        {"name": "yolo11m", "file": "yolo11m.pt"},
        {"name": "yolo11l", "file": "yolo11l.pt"},
        {"name": "yolo11x", "file": "yolo11x.pt"},
        # YOLO12 Serisi
        {"name": "yolo12n", "file": "yolo12n.pt"},
        {"name": "yolo12s", "file": "yolo12s.pt"},
        {"name": "yolo12m", "file": "yolo12m.pt"},
        {"name": "yolo12l", "file": "yolo12l.pt"},
        {"name": "yolo12x", "file": "yolo12x.pt"},
    ]
    print("\n" + "=" * 40)
    print("MODEL SEÇİMİ")
    print("=" * 40)
    print("\n📦 YOLOv10 Serisi:")
    for i, m in enumerate(models[:6], 1):
        status = "✅" if os.path.exists(m['file']) else "⬇️"
        print(f"  {i:2}. {m['name']:<12} {status}")
    print("\n📦 YOLO11 Serisi:")
    for i, m in enumerate(models[6:11], 7):
        status = "✅" if os.path.exists(m['file']) else "⬇️"
        print(f"  {i:2}. {m['name']:<12} {status}")
    print("\n📦 YOLO12 Serisi:")
    for i, m in enumerate(models[11:], 12):
        status = "✅" if os.path.exists(m['file']) else "⬇️"
        print(f"  {i:2}. {m['name']:<12} {status}")
    
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
    parser.add_argument('--checkpoints', type=str, default=None,
                        help='Checkpoint listesi, virgülle ayrılmış (örn: 25,50,100,200)')
    parser.add_argument('--model', type=str, default=None,
                        help='Model adı (örn: yolo12x) - interaktif menüyü atlar')
    parser.add_argument('--overwrite', action='store_true',
                        help='Sıfırdan başla (eski eğitimi sil)')
    return parser.parse_args()

def get_current_epoch(last_pt_path):
    """Checkpoint dosyasından mevcut epoch'u oku"""
    try:
        ckpt = torch.load(last_pt_path, map_location='cpu')
        return ckpt.get('epoch', -1) + 1
    except:
        return 0

def run_validation_and_report(model_path, yaml_path, cfg, model_name, epoch, result_dir, ckpt_dir, duration):
    """Validation çalıştır ve rapor kaydet"""
    print(f"\n{'='*50}")
    print(f"📊 CHECKPOINT {epoch} - Validation & Rapor")
    print(f"{'='*50}")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    val_model = YOLO(str(model_path))
    metrics = val_model.val(data=str(yaml_path), device=0, batch=cfg.batch_size, workers=0, verbose=False)
    
    save_report(model_name, metrics, duration, epoch, str(result_dir))
    
    # Checkpoint'i kaydet
    dest_pt = ckpt_dir / f"{model_name}_{epoch}ep.pt"
    shutil.copy(model_path, dest_pt)
    print(f"💾 Checkpoint Kaydedildi: {dest_pt}")
    
    del val_model
    gc.collect()
    torch.cuda.empty_cache()
    
    return metrics

def train_to_checkpoint(model, yaml_path, cfg, model_name, target_epoch, workers, is_first_run=False):
    """Belirli bir epoch'a kadar eğit"""
    print(f"\n🎯 Hedef Epoch: {target_epoch}")
    
    model.train(
        data=str(yaml_path),
        epochs=target_epoch,
        imgsz=cfg.imgsz,
        batch=cfg.batch_size,
        workers=workers,
        device=0,
        project="runs/detect",
        name=model_name,
        exist_ok=True,
        resume=not is_first_run,  # İlk çalıştırma değilse resume=True
        cache=cfg.cache_images if is_first_run else False,  # Cache sadece ilk seferde
        patience=0  # Early stopping kapalı (checkpoint'lere ulaşmak için)
    )

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
    print("YOLO EĞİTİM ARACI - CHECKPOINT MODU (V28)")
    print("="*50)
    
    # 1. Model Seçimi (CLI veya interaktif)
    if args.model:
        model_name = args.model
        print(f"📦 Model (CLI): {model_name}")
    else:
        model_name = select_model_menu()
    pt_file = f"{model_name}.pt"
    
    # 2. Checkpoint Listesi (CLI veya interaktif)
    if args.checkpoints:
        checkpoints = [int(x.strip()) for x in args.checkpoints.split(',')]
        print(f"🎯 Checkpoints (CLI): {checkpoints}")
    else:
        print("\n📋 Checkpoint epoch'larını virgülle girin")
        print("   Örnek: 25,50,100,200")
        checkpoint_input = input("Checkpoints: ").strip()
        try:
            checkpoints = [int(x.strip()) for x in checkpoint_input.split(',')]
        except:
            print("❌ Geçersiz giriş, varsayılan kullanılıyor: [25, 50, 100]")
            checkpoints = [25, 50, 100]
    
    # Checkpoint'leri sırala
    checkpoints = sorted(set(checkpoints))
    
    # 3. Mod Seçimi (CLI veya interaktif)
    if args.overwrite:
        action = "OVERWRITE"
    else:
        print("\n[R]esume (Devam Et) | [O]verwrite (Sıfırla)")
        mode = input("Seçim [R/o]: ").strip().upper()
        action = "OVERWRITE" if mode == 'O' else "RESUME"

    # Yolları belirle
    run_dir = Path("runs/detect") / model_name
    last_pt = run_dir / "weights" / "last.pt"
    
    workers = cfg.workers
    current_epoch = 0
    
    if action == "OVERWRITE":
        if run_dir.exists():
            try: shutil.rmtree(run_dir)
            except: pass
            print("🧹 Eski eğitim klasörü silindi.")
        current_weights = pt_file
        current_epoch = 0
    else:
        if last_pt.exists():
            current_epoch = get_current_epoch(last_pt)
            print(f"🔄 Kaldığı yer tespit edildi: Epoch {current_epoch}")
            current_weights = str(last_pt)
            force_patch_workers(last_pt)
        else:
            print("⚠️ Kayıtlı model bulunamadı, sıfırdan başlanıyor...")
            current_weights = pt_file
            current_epoch = 0
    
    # Zaten tamamlanmış checkpoint'leri atla
    remaining_checkpoints = [cp for cp in checkpoints if cp > current_epoch]
    
    if not remaining_checkpoints:
        print(f"\n✅ Tüm checkpoint'ler zaten tamamlanmış! (Mevcut: {current_epoch})")
        return
    
    print(f"\n{'='*50}")
    print(f"🚀 EĞİTİM PLANI")
    print(f"{'='*50}")
    print(f"📦 Model: {model_name.upper()}")
    print(f"📍 Başlangıç Epoch: {current_epoch}")
    print(f"🎯 Checkpoints: {remaining_checkpoints}")
    print(f"🏁 Final Epoch: {remaining_checkpoints[-1]}")
    print(f"⚡ Ayarlar: Cache={cfg.cache_images}, Workers={workers}, Batch={cfg.batch_size}")
    
    input("\n👉 Başlamak için ENTER...")
    
    if os.name == 'nt':
        torch.multiprocessing.set_sharing_strategy('file_system')

    gc.collect()
    torch.cuda.empty_cache()
    
    overall_start_time = time.time()
    
    try:
        # Model yükle
        print(f"\n📥 Model yükleniyor: {current_weights}...")
        try:
            model = YOLO(current_weights)
        except Exception as e:
            print(f"\n❌ HATA: Model dosyası '{current_weights}' bulunamadı veya indirilemedi.")
            print(f"🛑 Teknik Detay: {e}")
            sys.exit(1)

        # Her checkpoint için eğit ve rapor oluştur
        is_first_run = (current_epoch == 0)
        
        for i, target_epoch in enumerate(remaining_checkpoints):
            checkpoint_start_time = time.time()
            
            print(f"\n{'#'*50}")
            print(f"# CHECKPOINT {i+1}/{len(remaining_checkpoints)}: Epoch {target_epoch}")
            print(f"{'#'*50}")
            
            # Eğit
            train_to_checkpoint(model, yaml_path, cfg, model_name, target_epoch, workers, is_first_run)
            
            # Validation ve rapor
            final_last_pt = run_dir / "weights" / "last.pt"
            if final_last_pt.exists():
                checkpoint_duration = (time.time() - checkpoint_start_time) / 60.0
                total_duration = (time.time() - overall_start_time) / 60.0
                
                run_validation_and_report(
                    final_last_pt, yaml_path, cfg, model_name, 
                    target_epoch, result_dir, ckpt_dir, total_duration
                )
                
                # Modeli tekrar yükle (resume için)
                del model
                gc.collect()
                torch.cuda.empty_cache()
                model = YOLO(str(final_last_pt))
            
            is_first_run = False  # Artık resume modunda
            print(f"\n✅ Checkpoint {target_epoch} tamamlandı! ({checkpoint_duration:.1f} dk)")
        
        # Final özet
        total_duration = (time.time() - overall_start_time) / 60.0
        print(f"\n{'='*50}")
        print(f"🎉 TÜM CHECKPOINTS TAMAMLANDI!")
        print(f"{'='*50}")
        print(f"⏱️ Toplam Süre: {total_duration:.1f} dakika ({total_duration/60:.2f} saat)")
        print(f"📊 Raporlar: {result_dir / 'reports'}")
        print(f"💾 Modeller: {ckpt_dir}")

    except KeyboardInterrupt:
        print(f"\n\n⚠️ Eğitim kullanıcı tarafından durduruldu!")
        print(f"💡 Devam etmek için: python train.py --checkpoints {','.join(map(str, remaining_checkpoints))}")
    except Exception as e:
        print(f"\n❌ BEKLENMEYEN HATA: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
