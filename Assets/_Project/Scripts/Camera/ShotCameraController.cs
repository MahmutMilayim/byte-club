using UnityEngine;

public class ShotCameraController : MonoBehaviour
{
    [Header("Refs")]
    public Camera cam;
    public FrameSnapshotApplier applier;
    public MetersToWorldMapper mapper;

    [Header("Tuning")]
    public float height = 6.5f;
    public float backOffset = 10f;
    public float sideOffset = 2.0f;
    public float lookHeight = 1.0f;
    public bool useOrthographic = false;

    [Header("Auto")]
    public bool alignOnStart = false;

    private void Reset()
    {
        cam = Camera.main;
    }

    private void Start()
    {
        if (alignOnStart)
            Align();
    }

    [ContextMenu("Align Camera To Shooter")]
    public void Align()
    {
        if (cam == null || applier == null || mapper == null)
        {
            Debug.LogError("ShotCameraController: Missing refs (cam/applier/mapper).");
            return;
        }

        // 1) DTO'yu applier'dan almaya çalış
        FrameSnapshotDTO dto = null;

        // Eğer LastDTO doğru eklenmişse bunu kullanır
        dto = applier.LastDTO;

        // 2) Fallback: LastDTO yoksa JSON'dan tekrar parse et
        if (dto == null)
        {
            if (applier.frameJson == null)
            {
                Debug.LogWarning("ShotCameraController: No LastDTO and no frameJson assigned on applier.");
                return;
            }

            dto = JsonUtility.FromJson<FrameSnapshotDTO>(applier.frameJson.text);
        }

        if (dto == null || dto.players == null || dto.shooter == null)
        {
            Debug.LogWarning("ShotCameraController: No shooter snapshot loaded (dto/players/shooter missing).");
            return;
        }

        int shooterId = dto.shooter.playerId;

        if (!TryFindPlayerMeters(dto, shooterId, out var shooterMeters))
        {
            Debug.LogWarning($"ShotCameraController: Shooter id {shooterId} not found in players array.");
            return;
        }

        Vector3 shooterWorld = mapper.ToWorld(shooterMeters.x, shooterMeters.y);

        string targetGoal = (dto.shooter.targetGoal ?? "TOP").Trim().ToUpperInvariant();

        // goal center in meters:
        // width center = 34
        // length end = 0 or 105
        float widthCenter = 34f;
        float lengthEnd = (targetGoal == "BOTTOM") ? 0f : 105f;

        // mapper lengthAlongX=true ise yine ToWorld(width, length) doğru çalışacak

        
        Vector3 goalWorld = mapper.ToWorld(widthCenter, lengthEnd);
        Vector3 dir = (goalWorld - shooterWorld);
        dir.y = 0f;
        dir = dir.sqrMagnitude > 0.0001f ? dir.normalized : Vector3.forward;

        Vector3 right = Vector3.Cross(Vector3.up, dir).normalized;

        Vector3 camPos = shooterWorld - dir * backOffset + right * sideOffset;
        camPos.y = height;

        Vector3 lookTarget = Vector3.Lerp(shooterWorld, goalWorld, 0.65f);
        lookTarget.y = lookHeight;

        cam.transform.position = camPos;
        cam.transform.LookAt(lookTarget);

        if (useOrthographic)
        {
            cam.orthographic = true;
            cam.orthographicSize = 7.5f;
        }
        else
        {
            cam.orthographic = false;
        }
    }

    private bool TryFindPlayerMeters(FrameSnapshotDTO dto, int id, out (float x, float y) meters)
    {
        foreach (var p in dto.players)
        {
            if (p != null && p.id == id)
            {
                meters = (p.x, p.y);
                return true;
            }
        }

        meters = default;
        return false;
    }
}