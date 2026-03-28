using System.Collections.Generic;
using UnityEngine;

public class FrameSnapshotApplier : MonoBehaviour
{
    public event System.Action<FrameSnapshotDTO> OnSnapshotApplied;

    [Header("Input")]
    public TextAsset frameJson;
    public MetersToWorldMapper mapper;

    [Header("Prefabs")]
    public GameObject playerPrefab;
    public GameObject ballPrefab; // opsiyonel

    [Header("Runtime")]
    public Transform runtimeRoot;
    public bool hideMissingPlayers = true;

    [Header("Auto")]
    public bool applyOnStart = true;

    [Header("Markers")]
    public bool markShooter = true;
    public float shooterMarkerHeight = 1.8f;
    public float shooterMarkerScale = 0.25f;

    public FrameSnapshotDTO LastDTO { get; private set; }

    // id -> GO
    private readonly Dictionary<int, GameObject> _players = new();
    private GameObject _ballGO;

    private void Awake()
    {
        if (runtimeRoot == null)
        {
            var root = GameObject.Find("FootballRuntime") ?? new GameObject("FootballRuntime");
            runtimeRoot = root.transform;
        }
    }

    private void Start()
    {
        if (applyOnStart)
            ApplySnapshot();
    }

    public void ApplySnapshot()
    {
        if (frameJson == null || mapper == null || playerPrefab == null)
        {
            Debug.LogError("FrameSnapshotApplier: Missing references (frameJson/mapper/playerPrefab).");
            return;
        }

        var dto = JsonUtility.FromJson<FrameSnapshotDTO>(frameJson.text);
        LastDTO = dto;

        if (dto == null)
        {
            Debug.LogError("FrameSnapshotApplier: JSON parse failed.");
            return;
        }

        var seen = new HashSet<int>();

        // 0) Shooter (özel)
        int shooterId = -1;
        if (dto.shooter != null)
        {
            shooterId = dto.shooter.playerId;
            seen.Add(shooterId);

            var shooterGO = GetOrCreate(shooterId);
            shooterGO.name = $"SHOOTER_{shooterId}";
            shooterGO.transform.position = mapper.ToWorld(dto.shooter.x, dto.shooter.y);
            shooterGO.SetActive(true);

            if (markShooter) EnsureShooterMarker(shooterGO.transform);
        }

        // 0.1) Goalkeeper (özel, marker yok şimdilik)
        if (dto.goalkeeper != null)
        {
            int gkId = dto.goalkeeper.playerId;
            seen.Add(gkId);

            var gkGO = GetOrCreate(gkId);
            gkGO.name = $"GK_{gkId}";
            gkGO.transform.position = mapper.ToWorld(dto.goalkeeper.x, dto.goalkeeper.y);
            gkGO.SetActive(true);

            // GK ise shooter marker istemeyiz
            if (markShooter) RemoveShooterMarkerIfExists(gkGO.transform);
        }

        // 1) Normal players
        if (dto.players != null)
        {
            foreach (var p in dto.players)
            {
                if (p == null) continue;

                // shooter/gk players[] içinde yanlışlıkla gelirse çakışmasın
                if (p.id == shooterId) continue;
                if (dto.goalkeeper != null && p.id == dto.goalkeeper.playerId) continue;

                seen.Add(p.id);

                var go = GetOrCreate(p.id);
                go.name = $"P_{p.id}";
                go.transform.position = mapper.ToWorld(p.x, p.y);
                go.SetActive(true);

                if (markShooter) RemoveShooterMarkerIfExists(go.transform);
            }
        }

        // 2) hide missing
        if (hideMissingPlayers)
        {
            foreach (var kv in _players)
            {
                if (!seen.Contains(kv.Key) && kv.Value != null)
                {
                    kv.Value.SetActive(false);
                    if (markShooter) RemoveShooterMarkerIfExists(kv.Value.transform);
                }
            }
        }

                // 3) Ball
                if (dto.ball != null && dto.ball.visible && ballPrefab != null)
                {
                    if (_ballGO == null)
                    {
                        _ballGO = Instantiate(ballPrefab, runtimeRoot);
                        _ballGO.name = "Ball";
                    }

                    _ballGO.transform.position = mapper.ToWorld(dto.ball.x, dto.ball.y);
                    _ballGO.SetActive(true);
                }
                else
                {
                    if (_ballGO != null) _ballGO.SetActive(false);
                }

                ApplyFacing(dto);

                OnSnapshotApplied?.Invoke(dto);
                Debug.Log($"Applied snapshot frame {dto.frameIndex} (players: {(dto.players?.Length ?? 0)}, shooter: {shooterId}).");
    }

    private GameObject GetOrCreate(int id)
    {
        if (_players.TryGetValue(id, out var go) && go != null) return go;

        go = Instantiate(playerPrefab, runtimeRoot);
        _players[id] = go;
        return go;
    }

    private const string ShooterMarkerName = "__ShooterMarker";

    private void EnsureShooterMarker(Transform player)
    {
        var existing = player.Find(ShooterMarkerName);
        if (existing != null) return;

        var marker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        marker.name = ShooterMarkerName;
        marker.transform.SetParent(player, false);
        marker.transform.localPosition = new Vector3(0f, shooterMarkerHeight, 0f);
        marker.transform.localScale = Vector3.one * shooterMarkerScale;

        var col = marker.GetComponent<Collider>();
        if (col != null) Destroy(col);
    }

    private void RemoveShooterMarkerIfExists(Transform player)
    {
        var existing = player.Find(ShooterMarkerName);
        if (existing != null) Destroy(existing.gameObject);
    }

    private void ApplyFacing(FrameSnapshotDTO dto)
    {
        if (dto == null || dto.shooter == null) return;

        // Shooter world pos
        Vector3 shooterPos = mapper.ToWorld(dto.shooter.x, dto.shooter.y);

        // Goal center (targetGoal'a göre)
        string targetGoal = (dto.targetGoal ?? "TOP").Trim().ToUpperInvariant();

        float goalX = 34f; // field center
        float goalY = (targetGoal == "BOTTOM") ? 0f : 105f;

        Vector3 goalWorld = mapper.ToWorld(goalX, goalY);

        // 1️⃣ Shooter rotation → kaleye dön
        if (TryGetPlayerAnimationDriver(dto.shooter.playerId, out var shooterDriver))
        {
            shooterDriver.FaceTowards(goalWorld);
        }

        // 2️⃣ Diğer oyuncular → shooter’a dön
        foreach (var kv in _players)
        {
            int id = kv.Key;
            var go = kv.Value;

            if (go == null) continue;
            if (id == dto.shooter.playerId) continue;

            var driver = go.GetComponent<PlayerAnimationDriver>();
            if (driver != null)
            {
                driver.FaceTowards(shooterPos);
            }
        }
    }
    public bool TryGetPlayerAnimationDriver(int playerId, out PlayerAnimationDriver driver)
    {
        driver = null;

        if (!_players.TryGetValue(playerId, out var go) || go == null)
            return false;

        driver = go.GetComponent<PlayerAnimationDriver>();
        return driver != null;
    }
}