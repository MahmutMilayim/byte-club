#!/usr/bin/env bash
# Football Analytics — Dev Container Başlatma Scripti
#
# Bu script, workspace'deki src/ dizinini container içinde /work/scripts/ olarak
# mount ederek başlatır. Cursor'da yaptığın değişiklikler anında container'a yansır.
#
# Kullanım:
#   bash docker/run_dev.sh             → football_dev container'ı (yeniden) oluştur
#   bash docker/run_dev.sh --rm-only   → mevcut container'ı sadece sil
set -euo pipefail

CONTAINER_NAME="football_dev"
IMAGE="byteclub:version"
WORKSPACE="$(cd "$(dirname "$0")/.." && pwd)"

if [ "${1:-}" = "--rm-only" ]; then
  docker rm -f "$CONTAINER_NAME" 2>/dev/null && echo "Container silindi." || echo "Container zaten yoktu."
  exit 0
fi

# Varsa eski container'ı temizle
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Mevcut '$CONTAINER_NAME' durduruluyor ve siliniyor..."
  docker rm -f "$CONTAINER_NAME"
fi

echo "Container başlatılıyor: $CONTAINER_NAME"
echo "  Image    : $IMAGE"
echo "  Input    : $WORKSPACE/data → /input"
echo "  Output   : $WORKSPACE/outputs → /output"
echo "  Scripts  : $WORKSPACE/src → /work/scripts"
echo "  HostHome : $HOME → /hosthome (ro)"
echo ""

docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --shm-size=2g \
  -v "$WORKSPACE/data:/input" \
  -v "$WORKSPACE/outputs:/output" \
  -v "$WORKSPACE/src:/work/scripts" \
  -v "$HOME:/hosthome:ro" \
  "$IMAGE" \
  bash -c "tail -f /dev/null"

echo "Container '$CONTAINER_NAME' başlatıldı."
echo ""
echo "  Bağlanmak için : docker exec -it $CONTAINER_NAME bash"
echo "  Veya           : make shell   (proje dizininde)"
