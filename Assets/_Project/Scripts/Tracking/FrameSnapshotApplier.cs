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

    [ContextMenu("Apply Snapshot")]
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

        var seen = new HashSet<int>();

        // 1) Players: create if missing, then update position
        foreach (var p in dto.players)
        {
            seen.Add(p.id);

            if (!_players.TryGetValue(p.id, out var go) || go == null)
            {
                go = Instantiate(playerPrefab, runtimeRoot);
                go.name = $"P_{p.id}";
                _players[p.id] = go;
            }

            go.transform.position = mapper.ToWorld(p.x, p.y);
            go.SetActive(true);
        }

        // 2) Optional: hide players not in this snapshot
        if (hideMissingPlayers)
        {
            foreach (var kv in _players)
            {
                if (!seen.Contains(kv.Key) && kv.Value != null)
                    kv.Value.SetActive(false);
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

        Debug.Log($"Applied snapshot frame {dto.frameIndex} (players: {dto.players.Length}).");
    }
}