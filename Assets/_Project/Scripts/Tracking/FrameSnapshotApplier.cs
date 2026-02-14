using System.Collections.Generic;
using UnityEngine;

public class FrameSnapshotApplier : MonoBehaviour
{
    [Header("Input")]
    public TextAsset frameJson;                 // frame_0010.json
    public MetersToWorldMapper mapper;

    [Header("Prefabs")]
    public GameObject playerPrefab;             // SnowmanPlayer.prefab
    public GameObject ballPrefab;               // opsiyonel (yoksa null bırak)

    [Header("Runtime")]
    public Transform runtimeRoot;               // null ise otomatik oluşturur
    public bool hideMissingPlayers = true;      // frame’de olmayanları gizle
    [Header("Shooter")]
    public bool markShooter = true;
    public float shooterMarkerHeight = 1.8f;
    public float shooterMarkerScale = 0.25f;

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

    public void ApplySnapshot()
    {
        if (frameJson == null || mapper == null || playerPrefab == null)
        {
            Debug.LogError("FrameSnapshotApplier: Missing references (frameJson/mapper/playerPrefab).");
            return;
        }

        var dto = JsonUtility.FromJson<FrameSnapshotDTO>(frameJson.text);
        if (dto == null || dto.players == null)
        {
            Debug.LogError("FrameSnapshotApplier: JSON parse failed or players missing.");
            return;
        }

        // Shooter id (yoksa -1)
        int shooterId = (dto.shooter != null) ? dto.shooter.playerId : -1;

        var seen = new HashSet<int>();

        // 1) Players: create if missing, then update position
        foreach (var p in dto.players)
        {
            seen.Add(p.id);

            if (!_players.TryGetValue(p.id, out var go) || go == null)
            {
                go = Instantiate(playerPrefab, runtimeRoot);
                _players[p.id] = go;
            }

            bool isShooter = (p.id == shooterId);

            // İsim + shooter marker
            if (isShooter)
            {
                go.name = $"SHOOTER_{p.id}";
                if (markShooter) EnsureShooterMarker(go.transform);
            }
            else
            {
                go.name = $"P_{p.id}";
                if (markShooter) RemoveShooterMarkerIfExists(go.transform);
            }

            // Konum
            go.transform.position = mapper.ToWorld(p.x, p.y);
            go.SetActive(true);
        }

        // 2) Optional: hide players not in this snapshot
        if (hideMissingPlayers)
        {
            foreach (var kv in _players)
            {
                if (!seen.Contains(kv.Key) && kv.Value != null)
                {
                    kv.Value.SetActive(false);

                    // İstersen: gizlenenlerde marker da temizle (opsiyonel)
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

        Debug.Log($"Applied snapshot frame {dto.frameIndex} (players: {dto.players.Length}, shooter: {shooterId}).");
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

        // collider istemiyoruz
        var col = marker.GetComponent<Collider>();
        if (col != null) Destroy(col);
    }

    private void RemoveShooterMarkerIfExists(Transform player)
    {
        var existing = player.Find(ShooterMarkerName);
        if (existing != null) Destroy(existing.gameObject);
    }
}