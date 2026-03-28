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
    public float orthoSize = 7.5f;

    [Header("Auto")]
    public bool alignOnStart = false;

    private void Reset() => cam = Camera.main;

    private void OnEnable()
    {
        if (applier != null)
            applier.OnSnapshotApplied += HandleSnapshotApplied;
    }

    private void OnDisable()
    {
        if (applier != null)
            applier.OnSnapshotApplied -= HandleSnapshotApplied;
    }

    private void Start()
    {
        if (alignOnStart)
            Align();
    }

    private void HandleSnapshotApplied(FrameSnapshotDTO dto)
    {
        // dto kullanmak istersen: Align(dto)
        Align(dto);
    }

    [ContextMenu("Align Camera To Shooter")]
    public void Align()
    {
        Align(applier != null ? applier.LastDTO : null);
    }

    public void Align(FrameSnapshotDTO dto)
    {
        if (cam == null || applier == null || mapper == null)
        {
            Debug.LogError("ShotCameraController: Missing refs (cam/applier/mapper).");
            return;
        }

        // Fallback: DTO yoksa JSON parse et
        if (dto == null)
        {
            if (applier.frameJson == null)
            {
                Debug.LogWarning("ShotCameraController: No DTO and no frameJson assigned on applier.");
                return;
            }
            dto = JsonUtility.FromJson<FrameSnapshotDTO>(applier.frameJson.text);
        }

        if (dto == null || dto.shooter == null)
        {
            Debug.LogWarning("ShotCameraController: shooter missing.");
            return;
        }

        // Shooter (meters -> world)
        Vector3 shooterWorld = mapper.ToWorld(dto.shooter.x, dto.shooter.y);

        // Target goal (frame-level)
        string targetGoal = (dto.targetGoal ?? "TOP").Trim().ToUpperInvariant();

        // Kale merkezi (meters)
        float widthCenter = 34f;
        float lengthEnd = (targetGoal == "BOTTOM") ? 0f : 105f;

        Vector3 goalWorld = mapper.ToWorld(widthCenter, lengthEnd);

        // Yön
        Vector3 dir = (goalWorld - shooterWorld);
        dir.y = 0f;
        dir = dir.sqrMagnitude > 0.0001f ? dir.normalized : Vector3.forward;

        Vector3 right = Vector3.Cross(Vector3.up, dir).normalized;

        // Kamera konumu
        Vector3 camPos = shooterWorld - dir * backOffset + right * sideOffset;
        camPos.y = height;

        // Bakış hedefi
        Vector3 lookTarget = Vector3.Lerp(shooterWorld, goalWorld, 0.65f);
        lookTarget.y = lookHeight;

        cam.transform.position = camPos;
        cam.transform.LookAt(lookTarget);

        if (useOrthographic)
        {
            cam.orthographic = true;
            cam.orthographicSize = orthoSize;
        }
        else cam.orthographic = false;
    }
}