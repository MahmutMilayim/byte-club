import os, json, time
from typing import List, Dict, Any, Union
from dataclasses import asdict
from schemas.types import TracksData, FrameRecord, BallPosition 

def write_meta(out_dir: str,
               video_name: str,
               fps: float,
               width: int,
               height: int,
               conf: float = 0.1,
               iou: float = 0.5,
               model_name: str = "unknown",
               extra: Union[Dict[str, Any], None] = None) -> str:
    """
    Çıkış videosu ile birlikte meta bilgilerini JSON olarak döndürdük, ileride işimize test esnasında yarama ihtimaline akrşı ekledik.
    """
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "video": video_name,
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "conf": float(conf),
        "iou": float(iou),
        "model": model_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    if extra:
        meta.update(extra)
    path = os.path.join(out_dir, f"{os.path.splitext(video_name)[0]}.meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return path

def write_tracks_json(
    video_meta: Dict[str, Union[float, int]],
    team_info: Dict[str, int],
    frame_records: List[FrameRecord],
    cleaned_ball_track: List[BallPosition],
    output_dir: str,
    video_filename: str
) -> str:
    """
    Tüm işlenmiş ve temizlenmiş takip verilerini (oyuncu, top, meta) 
    TracksData şemasına uygun olarak JSON dosyasına yazar.
    """
    
    #TracksData objesini oluştur
    tracks_data_obj = TracksData(
        video=video_meta,
        teams=team_info,
        frames=frame_records,
        ball_track=cleaned_ball_track
    )
    
    #Dataclass'ı JSON serileştirmesi için standart Python dict'e dönüştür (asdict kullanılarak)
    data_to_write = asdict(tracks_data_obj)
    
    #Dosya yolunu oluştur (örneğin: test.mp4 -> output/test.tracks.json)
    base_name = os.path.splitext(video_filename)[0]
    output_filename = f"{base_name}.tracks.json"
    output_path = os.path.join(output_dir, output_filename)
    
    #JSON'a yaz
    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            # indent=4 kabul kriterlerinizden biriydi, okunabilirliği artırır.
            json.dump(data_to_write, f, ensure_ascii=False, indent=4)
        print(f"✅ Başarılı: Temizlenmiş takip verileri {output_path} konumuna yazıldı.")
        return output_path
    except Exception as e:
        print(f"❌ Hata: Tracks JSON dosyası yazılırken bir sorun oluştu: {e}")
        return ""