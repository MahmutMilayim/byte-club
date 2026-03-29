using System.Collections;
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

    private Coroutine _cameraMoveRoutine;

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

        AlignToShooterPosition(new Vector2(dto.shooter.x, dto.shooter.y), dto.targetGoal);
    }

    public void AlignToShooterPosition(Vector2 shooterMeters, string targetGoal)
    {
        if (cam == null || mapper == null)
        {
            Debug.LogError("ShotCameraController: Missing cam/mapper.");
            return;
        }

        ComputeCameraPose(shooterMeters, targetGoal, out Vector3 camPos, out Quaternion camRot);

        cam.transform.position = camPos;
        cam.transform.rotation = camRot;

        if (useOrthographic)
        {
            cam.orthographic = true;
            cam.orthographicSize = orthoSize;
        }
        else
        {
            cam.orthographic = false;
        }
    }

    public void SmoothAlignToShooterPosition(Vector2 shooterMeters, string targetGoal, float duration)
    {
        if (cam == null || mapper == null)
        {
            Debug.LogError("ShotCameraController: Missing cam/mapper.");
            return;
        }

        if (_cameraMoveRoutine != null)
            StopCoroutine(_cameraMoveRoutine);

        _cameraMoveRoutine = StartCoroutine(SmoothMoveRoutine(shooterMeters, targetGoal, duration));
    }

    private IEnumerator SmoothMoveRoutine(Vector2 shooterMeters, string targetGoal, float duration)
    {
        ComputeCameraPose(shooterMeters, targetGoal, out Vector3 targetPos, out Quaternion targetRot);

        Vector3 startPos = cam.transform.position;
        Quaternion startRot = cam.transform.rotation;

        if (useOrthographic)
        {
            cam.orthographic = true;
            cam.orthographicSize = orthoSize;
        }
        else
        {
            cam.orthographic = false;
        }

        if (duration <= 0.0001f)
        {
            cam.transform.position = targetPos;
            cam.transform.rotation = targetRot;
            _cameraMoveRoutine = null;
            yield break;
        }

        float elapsed = 0f;

        while (elapsed < duration)
        {
            float t = elapsed / duration;
            t = Mathf.SmoothStep(0f, 1f, t);

            cam.transform.position = Vector3.Lerp(startPos, targetPos, t);
            cam.transform.rotation = Quaternion.Slerp(startRot, targetRot, t);

            elapsed += Time.deltaTime;
            yield return null;
        }

        cam.transform.position = targetPos;
        cam.transform.rotation = targetRot;
        _cameraMoveRoutine = null;
    }

    private void ComputeCameraPose(Vector2 shooterMeters, string targetGoal, out Vector3 camPos, out Quaternion camRot)
    {
        Vector3 shooterWorld = mapper.ToWorld(shooterMeters.x, shooterMeters.y);

        string tg = (targetGoal ?? "TOP").Trim().ToUpperInvariant();

        float widthCenter = 34f;
        float lengthEnd = (tg == "BOTTOM") ? 0f : 105f;

        Vector3 goalWorld = mapper.ToWorld(widthCenter, lengthEnd);

        Vector3 dir = goalWorld - shooterWorld;
        dir.y = 0f;
        dir = dir.sqrMagnitude > 0.0001f ? dir.normalized : Vector3.forward;

        Vector3 right = Vector3.Cross(Vector3.up, dir).normalized;

        camPos = shooterWorld - dir * backOffset + right * sideOffset;
        camPos.y = height;

        Vector3 lookTarget = Vector3.Lerp(shooterWorld, goalWorld, 0.65f);
        lookTarget.y = lookHeight;

        camRot = Quaternion.LookRotation((lookTarget - camPos).normalized, Vector3.up);
    }
}