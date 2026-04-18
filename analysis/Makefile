CONTAINER ?= football_dev
SCRIPTS   := /work/scripts

.PHONY: help shell run run-optimal run-2-to-5 run-from-5 run-from-5-full run-from-5-reproject \
        stage2 stage3 stage4 stage4-review stage4-fix \
        stage5 stage5-only \
        stage7 \
        stage6-prep stage6-calibrate stage6-homography stage6-refine stage6-projection stage6-review stage6-finalize \
        stage6-v2-finalize stage6-v3-finalize stage6-v4-finalize \
        stage6-v2-refine stage6-v2-project stage6-v2-review \
        stage8 stage8-teamspot \
        stage9-summary \
        stage10-free-kicks \
        stage10-shots \
        rerun-from-projection \
        logs clean-outputs

# ──────────────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Football Analytics Pipeline — Makefile"
	@echo ""
	@echo "  make shell           → Container'a bash ile gir"
	@echo "  make run             → Tam pipeline'ı çalıştır"
	@echo ""
	@echo "  ── Tek Stage Çalıştırma ─────────────────────────────────────────"
	@echo "  make stage2          → Person tracking (YOLO + BotSORT)"
	@echo "  make stage3          → Gameplay filter"
	@echo "  make stage4          → Team clustering (chunked, 3 parça)"
	@echo "  make stage4-review   → Sadece review.mp4 (mevcut track_labels + gameplay)"
	@echo "  make stage4-fix      → Goalkeeper label düzeltme"
	@echo "  make stage5          → Ball tracking + review video (YOLO, jump-suppressed)"
	@echo ""
	@echo "  make stage7          → Possession (canonical)"
	@echo "  make stage6-prep     → Keyframe extraction (ffmpeg)"
	@echo "  make stage6-calibrate → PnLCalib batch kalibrasyon"
	@echo "  make stage6-homography → Motion-propagated homography map"
	@echo "  make stage6-refine → Evidence-driven homography rescue"
	@echo "  make stage6-projection → Unified pitch projection"
	@echo "  make stage6-review → Unified Stage 6 review overlay"
	@echo "  make stage6-finalize → Final overlay video + summary JSON"
	@echo "  make stage8-teamspot → resmi SoccerNet Team Ball Spotting inference"
	@echo "  make stage8          → spotting-first analytics (team spotting + analytics)"
	@echo "  make rerun-from-projection → Projection sonrasi kisa yeniden calistir"
	@echo ""
	@echo ""
	@echo "  ── Önerilen Tam Akış ────────────────────────────────────────────"
	@echo "  make run-optimal     → 2→son (YOLO top; keyframe/calib atlanır; keyframe_json dolu olmalı)"
	@echo "  make run             → Baştan sona (ffmpeg + PnLCalib dahil)"
	@echo "  make run-2-to-5      → Sadece Stage 2-5"
	@echo "  make run-from-5      → Stage 5 + 6-7-8-final; keyframe/calib ATLANIR"
	@echo "  make run-from-5-full → Aynı kuyruk ama ffmpeg + PnLCalib yeniden çalışır"
	@echo "  make run-from-5-reproject → run-from-5 ile aynı (geriye dönük isim)"
	@echo "  make stage5-only     → Sadece Stage 5 (ball_tracks); 6+ çalışmaz"
	@echo ""
	@echo "  make logs            → Container son logları"
	@echo "  make clean-outputs   → /output temizle (dikkatli!)"
	@echo ""

# ──────────────────────────────────────────────────────────────────────────────
shell:
	docker exec -it $(CONTAINER) bash

run:
	docker exec -it $(CONTAINER) bash $(SCRIPTS)/run_pipeline.sh

run-from-%:
	docker exec -it $(CONTAINER) bash $(SCRIPTS)/run_pipeline.sh --from $*

# Stage 5 yeniden + homography bank → projection → … ; keyframe_json (ve yedek stage6) hazır olmalı
run-from-5 run-from-5-reproject:
	docker exec -it $(CONTAINER) bash $(SCRIPTS)/run_pipeline.sh --from 5 --skip-keyframe-calib

# İlk kurulum veya sahayı baştan kalibre etmek için
run-from-5-full:
	docker exec -it $(CONTAINER) bash $(SCRIPTS)/run_pipeline.sh --from 5

run-to-%:
	docker exec -it $(CONTAINER) bash $(SCRIPTS)/run_pipeline.sh --to $*

#   make run-2-to-5          → Stage 2-5 (ball_tracks.json, YOLO)
#   make run-from-5          → 5 + 6-7-8-final (keyframe/calib atlanır)
run-2-to-5:
	docker exec -it $(CONTAINER) bash $(SCRIPTS)/run_pipeline.sh --from 2 --to 5

# YOLO top (5b yok): 2→son tek komut; keyframe_json dolu olmalı (--skip-keyframe-calib)
run-optimal:
	@echo "══ Stage 2 → son | YOLO top | ffmpeg+PnLCalib atlanır (keyframe_json gerekli) ══"
	docker exec -it $(CONTAINER) bash $(SCRIPTS)/run_pipeline.sh --from 2 --skip-keyframe-calib

logs:
	docker logs --tail=80 $(CONTAINER)

# ──────────────────────────────────────────────────────────────────────────────
stage2:
	docker exec $(CONTAINER) python $(SCRIPTS)/stage2_tracking/person_track.py

stage3:
	docker exec $(CONTAINER) python $(SCRIPTS)/stage3_filter/gameplay_filter.py

stage4:
	docker exec $(CONTAINER) python $(SCRIPTS)/stage4_clustering/clustering.py

stage4-review:
	docker exec $(CONTAINER) python $(SCRIPTS)/stage4_clustering/clustering.py --review-only

stage4-fix:
	docker exec $(CONTAINER) python $(SCRIPTS)/stage4_clustering/fix_goalkeepers.py

stage5:
	docker exec $(CONTAINER) python $(SCRIPTS)/stage5_ball/ball_tracking.py

# Sadece top izi — keyframe / kalibrasyon / projection yok
stage5-only:
	docker exec $(CONTAINER) bash $(SCRIPTS)/run_pipeline.sh --from 5 --to 5

stage7:
	docker exec $(CONTAINER) python $(SCRIPTS)/stage7_possession/possession.py

stage6-prep:
	docker exec $(CONTAINER) bash -c '\
	  mkdir -p /output/stage6_field/{keyframes,keyframe_json,keyframe_projected,debug,logs,single_frame_runs} && \
	  ffmpeg -y -i /output/stage3_filter/gameplay.mp4 \
	    -vf "select='"'"'not(mod(n,5))'"'"'" -vsync vfr \
	    /output/stage6_field/keyframes/frame_%06d.png'

stage6-calibrate:
	docker exec $(CONTAINER) bash -c '\
	  cd /work/PnLCalib && \
	  PYTHONPATH=/work/PnLCalib python $(SCRIPTS)/stage6_field/calibrate.py'

stage6-homography:
	docker exec $(CONTAINER) python $(SCRIPTS)/stage6_field/homography_motion.py

stage6-refine:
	docker exec $(CONTAINER) python $(SCRIPTS)/stage6_field/refine_homography.py

stage6-projection:
	docker exec $(CONTAINER) python $(SCRIPTS)/stage6_field/projection.py

stage6-review:
	docker exec $(CONTAINER) python $(SCRIPTS)/stage6_field/review.py

stage6-finalize:
	docker exec $(CONTAINER) python $(SCRIPTS)/stage6_field/finalize.py

stage6-v2-finalize:
	docker exec $(CONTAINER) python $(SCRIPTS)/stage6_field/finalize.py \
	  --projection-json /output/stage6_field/projection.json \
	  --possession-json /output/stage7_possession/possession.json \
	  --hmap-json /output/stage6_field/homography_map_refined.json \
	  --out-summary /output/final_v2/summary.json \
	  --out-video /output/final_v2/overlay.mp4

stage6-v3-finalize:
	docker exec $(CONTAINER) python $(SCRIPTS)/stage6_field/finalize.py \
	  --projection-json /output/stage6_field/projection.json \
	  --possession-json /output/stage7_possession/possession.json \
	  --hmap-json /output/stage6_field/homography_map_refined.json \
	  --out-summary /output/final_v3/summary.json \
	  --out-video /output/final_v3/overlay.mp4

stage6-v4-finalize:
	docker exec $(CONTAINER) python $(SCRIPTS)/stage6_field/finalize.py \
	  --projection-json /output/stage6_field/projection.json \
	  --possession-json /output/stage7_possession/possession.json \
	  --hmap-json /output/stage6_field/homography_map_refined.json \
	  --out-summary /output/final_v4/summary.json \
	  --out-video /output/final_v4/overlay.mp4

stage6-v2-refine:
	docker exec $(CONTAINER) python $(SCRIPTS)/stage6_field/refine_homography.py

stage6-v2-project:
	docker exec $(CONTAINER) python $(SCRIPTS)/stage6_field/projection.py

stage6-v2-review:
	docker exec $(CONTAINER) python $(SCRIPTS)/stage6_field/review.py

# Kısmi yeniden çalıştırma (projeksiyon + kaleci fix + merge + possession + analytics + final)
rerun-from-projection:
	docker exec -it $(CONTAINER) bash -c '\
	  python $(SCRIPTS)/stage6_field/projection.py && \
	  python $(SCRIPTS)/stage4_clustering/fix_goalkeepers.py && \
	  python $(SCRIPTS)/stage4_clustering/merge_tracks.py && \
	  python $(SCRIPTS)/stage6_field/projection.py && \
	  mkdir -p /output/stage8_analytics && \
	  python $(SCRIPTS)/stage8_analytics/run_team_spotting.py && \
	  python $(SCRIPTS)/stage7_possession/possession.py && \
	  python $(SCRIPTS)/stage8_analytics/analytics.py && \
	  python $(SCRIPTS)/stage6_field/finalize.py && \
	  echo "TAMAMLANDI"'

stage8:
	docker exec $(CONTAINER) bash -lc 'mkdir -p /output/stage8_analytics && python $(SCRIPTS)/stage8_analytics/run_team_spotting.py && python $(SCRIPTS)/stage8_analytics/analytics.py'

stage8-teamspot:
	docker exec $(CONTAINER) bash -lc 'mkdir -p /output/stage8_analytics && python $(SCRIPTS)/stage8_analytics/run_team_spotting.py'

# ──────────────────────────────────────────────────────────────────────────────
clean-outputs:
	@echo "UYARI: /output altındaki tüm stage klasörleri silinecek!"
	@read -p "Emin misin? [y/N] " ans; [ "$$ans" = "y" ] || exit 1
	docker exec $(CONTAINER) bash -c '\
	  rm -rf /output/stage2_tracking \
	         /output/stage2b_pose \
	         /output/stage3_filter \
	         /output/stage4_clustering \
	         /output/stage5_ball \
	         /output/stage5b_tracknet \
	         /output/stage7_possession \
	         /output/stage6_field \
	         /output/final \
	         /output/stage8_analytics && \
	  echo "Temizlendi."'

stage9-summary:
	docker exec $(CONTAINER) bash -lc 'mkdir -p /output/stage9_summary && python $(SCRIPTS)/stage9_summary/summary_builder.py'

stage10-free-kicks:
	docker exec $(CONTAINER) bash -lc 'mkdir -p /output/stage10_setpiece_inputs && python $(SCRIPTS)/stage10_setpiece_inputs/export_free_kick_inputs.py'

stage10-shots:
	docker exec $(CONTAINER) bash -lc 'mkdir -p /output/stage10_setpiece_inputs && python $(SCRIPTS)/stage10_setpiece_inputs/export_shot_inputs_pnlcalib.py'
