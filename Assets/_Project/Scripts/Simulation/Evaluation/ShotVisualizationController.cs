using System.Collections.Generic;
using UnityEngine;

[DisallowMultipleComponent]
public class ShotVisualizationController : MonoBehaviour
{
    [Header("Refs")]
    public FrameSnapshotApplier applier;
    public MetersToWorldMapper mapper;
    public BestPassEvaluator bestPassEvaluator;

    [Header("Goal (meters)")]
    public float goalWidthMeters = 7.32f;
    public float fieldLengthMeters = 105f;
    public float fieldWidthMeters = 68f;

    [Header("Block radii (meters)")]
    public float playerBlockRadius = 1.25f;
    public float goalkeeperBlockRadius = 2.0f;

    [Header("Shooter source")]
    public bool useDecisionFinalShooter = true;

    [Header("Goal sampling")]
    [Range(10, 120)] public int goalSamples = 48;

    [Header("Meshes")]
    public bool drawFullGoalMesh = true;
    public Material fullGoalMaterial;

    public bool drawOpenGoalMesh = true;
    public Material openGoalMaterial;

    public float fullFillYOffset = 0.05f;
    public float openFillYOffset = 0.07f;

    [Header("Debug drawing")]
    public bool drawGizmos = true;
    public Color shooterToPostColor = Color.yellow;
    public Color blockerCircleColor = new Color(1f, 0.2f, 0.2f, 1f);
    public Color openSegmentColor = new Color(1f, 0.55f, 0.1f, 1f);
    public Color blockedSegmentColor = new Color(1f, 0.2f, 0.2f, 1f);

    [Header("Auto")]
    public bool rebuildOnSnapshotApplied = true;

    private MeshFilter _fullMf;
    private MeshRenderer _fullMr;
    private MeshFilter _openMf;
    private MeshRenderer _openMr;

    private FrameSnapshotDTO _last;
    private Computation _lastComp;

    private struct Blocker
    {
        public Vector3 posWorld;
        public float radiusMeters;
        public string label;
    }

    private struct GoalSegment
    {
        public float t0;
        public float t1;

        public GoalSegment(float t0, float t1)
        {
            this.t0 = t0;
            this.t1 = t1;
        }
    }

    private struct Computation
    {
        public int effectiveShooterId;
        public string effectiveShooterTeamId;

        public Vector2 shooterM;
        public Vector3 shooterW;

        public Vector3 goalLeftW;
        public Vector3 goalRightW;

        public List<Blocker> blockers;
        public List<GoalSegment> openSegments;
        public List<GoalSegment> blockedSegments;
    }

    private void Awake()
    {
        EnsureMeshObjects();
    }

    private void OnEnable()
    {
        EnsureMeshObjects();

        if (rebuildOnSnapshotApplied && applier != null)
            applier.OnSnapshotApplied += HandleSnapshotApplied;
    }

    private void OnDisable()
    {
        if (rebuildOnSnapshotApplied && applier != null)
            applier.OnSnapshotApplied -= HandleSnapshotApplied;
    }

    private void HandleSnapshotApplied(FrameSnapshotDTO dto)
    {
        _last = dto;
        Rebuild();
    }

    [ContextMenu("Rebuild Visualization")]
    public void Rebuild()
    {
        if (applier == null || mapper == null)
        {
            Debug.LogWarning("ShotVisualizationController: Missing applier/mapper.");
            return;
        }

        var dto = applier.LastDTO;
        if (dto == null && applier.frameJson != null)
            dto = JsonUtility.FromJson<FrameSnapshotDTO>(applier.frameJson.text);

        if (dto == null || dto.shooter == null)
        {
            Debug.LogWarning("ShotVisualizationController: Missing DTO/shooter.");
            return;
        }

        _last = dto;
        _lastComp = Compute(dto);

        if (drawFullGoalMesh)
            BuildFullGoalMesh(_lastComp);
        else
            ClearFullMesh();

        if (drawOpenGoalMesh)
            BuildOpenGoalMesh(_lastComp);
        else
            ClearOpenMesh();
    }

    private Computation Compute(FrameSnapshotDTO dto)
    {
        var comp = new Computation
        {
            blockers = new List<Blocker>(32),
            openSegments = new List<GoalSegment>(),
            blockedSegments = new List<GoalSegment>()
        };

        ResolveEffectiveShooter(dto, out int shooterId, out string shooterTeamId, out Vector2 shooterMeters);

        comp.effectiveShooterId = shooterId;
        comp.effectiveShooterTeamId = shooterTeamId;
        comp.shooterM = shooterMeters;
        comp.shooterW = mapper.ToWorld(shooterMeters.x, shooterMeters.y);

        string targetGoal = (dto.targetGoal ?? "TOP").Trim().ToUpperInvariant();
        float goalY = (targetGoal == "BOTTOM") ? 0f : fieldLengthMeters;

        float goalCenterX = fieldWidthMeters * 0.5f;
        float halfGoal = goalWidthMeters * 0.5f;

        float leftX = goalCenterX - halfGoal;
        float rightX = goalCenterX + halfGoal;

        comp.goalLeftW = mapper.ToWorld(leftX, goalY);
        comp.goalRightW = mapper.ToWorld(rightX, goalY);

        // shooter hariç herkes blocker
        if (dto.players != null)
        {
            foreach (var p in dto.players)
            {
                if (p == null) continue;
                if (p.id == shooterId) continue;

                comp.blockers.Add(new Blocker
                {
                    posWorld = mapper.ToWorld(p.x, p.y),
                    radiusMeters = playerBlockRadius,
                    label = $"P_{p.id}"
                });
            }
        }

        if (dto.goalkeeper != null && dto.goalkeeper.playerId != shooterId)
        {
            comp.blockers.Add(new Blocker
            {
                posWorld = mapper.ToWorld(dto.goalkeeper.x, dto.goalkeeper.y),
                radiusMeters = goalkeeperBlockRadius,
                label = $"GK_{dto.goalkeeper.playerId}"
            });
        }

        BuildGoalSegmentsBySampling(comp, out comp.openSegments, out comp.blockedSegments);

        return comp;
    }

    private void ResolveEffectiveShooter(FrameSnapshotDTO dto, out int shooterId, out string shooterTeamId, out Vector2 shooterMeters)
    {
        shooterId = dto.shooter.playerId;
        shooterTeamId = dto.shooter.teamId;
        shooterMeters = new Vector2(dto.shooter.x, dto.shooter.y);

        if (!useDecisionFinalShooter || bestPassEvaluator == null || bestPassEvaluator.LastResult == null)
            return;

        var decision = bestPassEvaluator.LastResult;
        int finalShooterId = decision.finalShooterId;

        if (finalShooterId == dto.shooter.playerId)
            return;

        if (dto.players != null)
        {
            foreach (var p in dto.players)
            {
                if (p == null) continue;
                if (p.id != finalShooterId) continue;

                shooterId = p.id;
                shooterTeamId = p.teamId;
                shooterMeters = new Vector2(p.x, p.y);
                return;
            }
        }

        if (dto.goalkeeper != null && dto.goalkeeper.playerId == finalShooterId)
        {
            shooterId = dto.goalkeeper.playerId;
            shooterTeamId = dto.goalkeeper.teamId;
            shooterMeters = new Vector2(dto.goalkeeper.x, dto.goalkeeper.y);
        }
    }

    private void BuildGoalSegmentsBySampling(Computation comp, out List<GoalSegment> openSegments, out List<GoalSegment> blockedSegments)
    {
        openSegments = new List<GoalSegment>();
        blockedSegments = new List<GoalSegment>();

        if (goalSamples < 2)
            return;

        bool[] openFlags = new bool[goalSamples];

        for (int i = 0; i < goalSamples; i++)
        {
            float t = (i + 0.5f) / goalSamples;
            Vector3 goalPoint = Vector3.Lerp(comp.goalLeftW, comp.goalRightW, t);

            bool blocked = IsBlocked(comp.shooterW, goalPoint, comp.blockers);
            openFlags[i] = !blocked;
        }

        BuildSegmentsFromFlags(openFlags, true, openSegments);
        BuildSegmentsFromFlags(openFlags, false, blockedSegments);
    }

    private void BuildSegmentsFromFlags(bool[] flags, bool targetValue, List<GoalSegment> segments)
    {
        int n = flags.Length;
        int start = -1;

        for (int i = 0; i < n; i++)
        {
            if (flags[i] == targetValue)
            {
                if (start == -1) start = i;
            }
            else
            {
                if (start != -1)
                {
                    float t0 = start / (float)n;
                    float t1 = i / (float)n;
                    segments.Add(new GoalSegment(t0, t1));
                    start = -1;
                }
            }
        }

        if (start != -1)
        {
            float t0 = start / (float)n;
            float t1 = 1f;
            segments.Add(new GoalSegment(t0, t1));
        }
    }

    private bool IsBlocked(Vector3 shooterW, Vector3 goalPointW, List<Blocker> blockers)
    {
        Vector2 a = new Vector2(shooterW.x, shooterW.z);
        Vector2 b = new Vector2(goalPointW.x, goalPointW.z);

        for (int i = 0; i < blockers.Count; i++)
        {
            Vector2 c = new Vector2(blockers[i].posWorld.x, blockers[i].posWorld.z);
            float r = blockers[i].radiusMeters;

            if (SegmentIntersectsCircle(a, b, c, r))
                return true;
        }

        return false;
    }

    private bool SegmentIntersectsCircle(Vector2 a, Vector2 b, Vector2 c, float r)
    {
        Vector2 ab = b - a;
        float ab2 = ab.sqrMagnitude;

        if (ab2 < 1e-8f)
            return (a - c).sqrMagnitude <= r * r;

        float t = Vector2.Dot(c - a, ab) / ab2;
        t = Mathf.Clamp01(t);

        Vector2 p = a + t * ab;
        return (p - c).sqrMagnitude <= r * r;
    }

    // -----------------------------
    // Mesh handling
    // -----------------------------
    private void EnsureMeshObjects()
    {
        EnsureChildMesh(ref _fullMf, ref _fullMr, "__FullGoalMesh", fullGoalMaterial);
        EnsureChildMesh(ref _openMf, ref _openMr, "__OpenGoalMesh", openGoalMaterial);
    }

    private void EnsureChildMesh(ref MeshFilter mf, ref MeshRenderer mr, string childName, Material mat)
    {
        Transform child = transform.Find(childName);
        if (child == null)
        {
            GameObject go = new GameObject(childName);
            go.transform.SetParent(transform, false);
            child = go.transform;
        }

        mf = child.GetComponent<MeshFilter>();
        if (mf == null) mf = child.gameObject.AddComponent<MeshFilter>();

        mr = child.GetComponent<MeshRenderer>();
        if (mr == null) mr = child.gameObject.AddComponent<MeshRenderer>();

        if (mat != null)
            mr.sharedMaterial = mat;
    }

    private void ClearFullMesh()
    {
        if (_fullMf != null) _fullMf.sharedMesh = null;
    }

    private void ClearOpenMesh()
    {
        if (_openMf != null) _openMf.sharedMesh = null;
    }

    private void BuildFullGoalMesh(Computation comp)
    {
        EnsureMeshObjects();

        if (fullGoalMaterial != null && _fullMr != null)
            _fullMr.sharedMaterial = fullGoalMaterial;

        Mesh mesh = new Mesh();
        mesh.name = "FullGoalConeMesh";

        Vector3 shooter = comp.shooterW;
        shooter.y = fullFillYOffset;

        Vector3 left = comp.goalLeftW;
        left.y = fullFillYOffset;

        Vector3 right = comp.goalRightW;
        right.y = fullFillYOffset;

        var verts = new List<Vector3> { shooter, left, right };
        var tris = new List<int> { 0, 2, 1 };

        mesh.SetVertices(verts);
        mesh.SetTriangles(tris, 0);
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();

        _fullMf.sharedMesh = mesh;
    }

    private void BuildOpenGoalMesh(Computation comp)
    {
        EnsureMeshObjects();

        if (openGoalMaterial != null && _openMr != null)
            _openMr.sharedMaterial = openGoalMaterial;

        var verts = new List<Vector3>(256);
        var tris = new List<int>(512);

        Vector3 shooter = comp.shooterW;
        shooter.y = openFillYOffset;

        for (int i = 0; i < comp.openSegments.Count; i++)
        {
            var seg = comp.openSegments[i];

            Vector3 p0 = Vector3.Lerp(comp.goalLeftW, comp.goalRightW, seg.t0);
            Vector3 p1 = Vector3.Lerp(comp.goalLeftW, comp.goalRightW, seg.t1);

            p0.y = openFillYOffset;
            p1.y = openFillYOffset;

            int baseIndex = verts.Count;
            verts.Add(shooter);
            verts.Add(p0);
            verts.Add(p1);

            tris.Add(baseIndex + 0);
            tris.Add(baseIndex + 2);
            tris.Add(baseIndex + 1);
        }

        Mesh mesh = new Mesh();
        mesh.name = "OpenGoalSegmentsMesh";
        mesh.SetVertices(verts);
        mesh.SetTriangles(tris, 0);
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();

        _openMf.sharedMesh = mesh;
    }

    // -----------------------------
    // Gizmos
    // -----------------------------
    private void OnDrawGizmos()
    {
        if (!drawGizmos) return;
        if (_last == null) return;
        if (mapper == null) return;

        if (_lastComp.blockers == null)
            _lastComp = Compute(_last);

        var comp = _lastComp;

        Gizmos.color = shooterToPostColor;
        Gizmos.DrawLine(comp.shooterW, comp.goalLeftW);
        Gizmos.DrawLine(comp.shooterW, comp.goalRightW);

        Gizmos.color = blockerCircleColor;
        foreach (var b in comp.blockers)
            DrawCircleXZ(b.posWorld, b.radiusMeters, 24);

        DrawGoalSegments(comp.openSegments, comp, openSegmentColor, openFillYOffset + 0.02f);
        DrawGoalSegments(comp.blockedSegments, comp, blockedSegmentColor, openFillYOffset + 0.01f);
    }

    private void DrawGoalSegments(List<GoalSegment> segments, Computation comp, Color c, float y)
    {
        if (segments == null) return;

        Gizmos.color = c;

        for (int i = 0; i < segments.Count; i++)
        {
            Vector3 p0 = Vector3.Lerp(comp.goalLeftW, comp.goalRightW, segments[i].t0);
            Vector3 p1 = Vector3.Lerp(comp.goalLeftW, comp.goalRightW, segments[i].t1);

            p0.y = y;
            p1.y = y;

            Gizmos.DrawLine(p0, p1);
            Gizmos.DrawLine(comp.shooterW, p0);
            Gizmos.DrawLine(comp.shooterW, p1);
        }
    }

    private static void DrawCircleXZ(Vector3 center, float radius, int segments)
    {
        center.y += 0.02f;
        float step = (2f * Mathf.PI) / Mathf.Max(8, segments);

        Vector3 prev = center + new Vector3(Mathf.Sin(0f), 0f, Mathf.Cos(0f)) * radius;
        for (int i = 1; i <= segments; i++)
        {
            float a = i * step;
            Vector3 cur = center + new Vector3(Mathf.Sin(a), 0f, Mathf.Cos(a)) * radius;
            Gizmos.DrawLine(prev, cur);
            prev = cur;
        }
    }
    public float ComputeShotProbability()
    {
        if (_lastComp.openSegments == null)
            return 0f;

        float openLength = 0f;

        foreach (var seg in _lastComp.openSegments)
        {
            openLength += (seg.t1 - seg.t0);
        }

        // zaten total = 1.0 (normalize edilmiş)
        return openLength;
    }
}
