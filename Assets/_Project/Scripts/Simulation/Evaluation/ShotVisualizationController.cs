using System;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Visualizes shot "openness":
/// - Lines from shooter to goal posts
/// - Blocking circles around opponents (and GK)
/// - Filled mesh wedges for OPEN regions of the goal cone (from shooter POV)
///
/// Assumes meters domain:
///   x: 0..68  (field width)
///   y: 0..105 (field length)
/// and mapper.ToWorld(x,y) maps to Unity XZ plane.
/// </summary>
[DisallowMultipleComponent]
public class ShotVisualizationController : MonoBehaviour
{
    [Header("Refs")]
    public FrameSnapshotApplier applier;
    public MetersToWorldMapper mapper;

    [Header("Goal (meters)")]
    [Tooltip("Official goal width is 7.32m. Use 7.32 unless you intentionally change.")]
    public float goalWidthMeters = 7.32f;

    [Tooltip("Goal line length coordinate in meters will be 0 or 105 depending on targetGoal.")]
    public float fieldLengthMeters = 105f;

    [Tooltip("Field width is 68m.")]
    public float fieldWidthMeters = 68f;

    [Header("Block radii (meters)")]
    [Tooltip("How much an outfield opponent blocks (radius around their position).")]
    public float opponentBlockRadius = 1.25f;

    [Tooltip("Goalkeeper blocks more (hands reach).")]
    public float goalkeeperBlockRadius = 2.0f;

    [Header("Filtering")]
    [Tooltip("Only players whose teamId != shooter.teamId are treated as blockers.")]
    public bool onlyOpponentsBlock = true;

    [Tooltip("If true, teammates do not block the shot cone.")]
    public bool ignoreTeammates = true;

    [Header("Open-area mesh")]
    public bool drawOpenFillMesh = true;
    public Material openFillMaterial;
    [Range(8, 256)] public int arcSegments = 64;
    [Tooltip("Distance from shooter used to draw the filled wedge (purely visual).")]
    public float fillDistance = 25f;
    [Tooltip("Lift the mesh slightly to avoid z-fighting.")]
    public float fillYOffset = 0.06f;

    [Header("Debug drawing")]
    public bool drawGizmos = true;
    public Color shooterToPostColor = Color.yellow;
    public Color blockerCircleColor = new Color(1f, 0.2f, 0.2f, 1f);
    public Color openArcColor = new Color(0.2f, 1f, 0.2f, 1f);
    public Color blockedArcColor = new Color(1f, 0.4f, 0.1f, 1f);

    [Header("Auto")]
    public bool rebuildOnSnapshotApplied = true;

    // --- internal ---
    private MeshFilter _mf;
    private MeshRenderer _mr;

    private FrameSnapshotDTO _last;
    private Computation _lastComp;

    private const float EPS = 1e-5f;

    // Interval struct: avoids tuple name issues completely
    private struct Interval
    {
        public float a0;
        public float a1;
        public Interval(float a0, float a1) { this.a0 = a0; this.a1 = a1; }
    }

    private struct Blocker
    {
        public Vector3 posWorld; // XZ meaningful
        public float radiusMeters;
        public string label;
    }

    private struct Computation
    {
        public Vector3 shooterW;
        public Vector3 goalLeftW;
        public Vector3 goalRightW;
        public Interval goalCone;              // [a0,a1] unwrapped
        public List<Interval> blocked;         // in goalCone space
        public List<Interval> open;            // complement in goalCone
        public List<Blocker> blockers;
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

        // Prefer LastDTO, fallback parse
        var dto = applier.LastDTO;
        if (dto == null && applier.frameJson != null)
            dto = JsonUtility.FromJson<FrameSnapshotDTO>(applier.frameJson.text);

        if (dto == null)
        {
            Debug.LogWarning("ShotVisualizationController: No DTO to visualize.");
            return;
        }

        if (dto.shooter == null)
        {
            Debug.LogWarning("ShotVisualizationController: DTO.shooter is null.");
            return;
        }

        _last = dto;
        _lastComp = Compute(dto);

        if (drawOpenFillMesh)
            BuildOpenMesh(_lastComp);
        else
            ClearMesh();
    }

    // -----------------------------
    // Core geometry
    // -----------------------------
    private Computation Compute(FrameSnapshotDTO dto)
    {
        var comp = new Computation
        {
            blocked = new List<Interval>(32),
            open = new List<Interval>(32),
            blockers = new List<Blocker>(32)
        };

        // Shooter world
        comp.shooterW = mapper.ToWorld(dto.shooter.x, dto.shooter.y);

        // Target goal: "TOP" => lengthEnd=105, "BOTTOM" => lengthEnd=0
        string targetGoal = (dto.targetGoal ?? "TOP").Trim().ToUpperInvariant();
        float lengthEnd = (targetGoal == "BOTTOM") ? 0f : fieldLengthMeters;

        float goalCenterX = fieldWidthMeters * 0.5f; // 34
        float halfGoal = goalWidthMeters * 0.5f;

        // Goal posts in meters
        float leftX = goalCenterX - halfGoal;
        float rightX = goalCenterX + halfGoal;
        float goalY = lengthEnd;

        comp.goalLeftW = mapper.ToWorld(leftX, goalY);
        comp.goalRightW = mapper.ToWorld(rightX, goalY);

        // Goal cone angles around shooter in XZ plane
        float aL = AngleXZ(comp.shooterW, comp.goalLeftW);
        float aR = AngleXZ(comp.shooterW, comp.goalRightW);

        // Normalize goal interval (unwrapped so that a1 >= a0)
        comp.goalCone = NormalizeInterval(aL, aR);

        // Build blocker list: opponents in players + explicit GK
        string shooterTeam = dto.shooter.teamId;

        // players[]
        if (dto.players != null)
        {
            foreach (var p in dto.players)
            {
                if (p == null) continue;

                // shooter might also exist in players list; ignore if same id
                if (p.id == dto.shooter.playerId) continue;

                if (onlyOpponentsBlock && ignoreTeammates)
                {
                    if (!string.IsNullOrEmpty(shooterTeam) && p.teamId == shooterTeam)
                        continue;
                }

                // If onlyOpponentsBlock true, but shooterTeam missing, we still include them (best effort).
                if (onlyOpponentsBlock && !string.IsNullOrEmpty(shooterTeam) && p.teamId == shooterTeam)
                    continue;

                var w = mapper.ToWorld(p.x, p.y);
                comp.blockers.Add(new Blocker
                {
                    posWorld = w,
                    radiusMeters = opponentBlockRadius,
                    label = $"P_{p.id}"
                });
            }
        }

        // explicit GK (as requested: GK separate like shooter)
        if (dto.goalkeeper != null)
        {
            // If GK is on same team as shooter, then it's likely wrong input (GK should be opponent).
            // Still include, but you can fix upstream.
            var gkw = mapper.ToWorld(dto.goalkeeper.x, dto.goalkeeper.y);
            comp.blockers.Add(new Blocker
            {
                posWorld = gkw,
                radiusMeters = goalkeeperBlockRadius,
                label = $"GK_{dto.goalkeeper.playerId}"
            });
        }

        // Convert blockers to angular intervals and clip to goal cone
        foreach (var b in comp.blockers)
        {
            if (TryBlockInterval(comp.shooterW, b.posWorld, b.radiusMeters, comp.goalCone, out var clipped))
                comp.blocked.Add(clipped);
        }

        comp.blocked = Merge(comp.blocked);
        comp.open = ComplementIntervals(comp.goalCone, comp.blocked);

        return comp;
    }

    private static float AngleXZ(Vector3 from, Vector3 to)
    {
        Vector3 d = to - from;
        d.y = 0f;
        if (d.sqrMagnitude < EPS) return 0f;
        // atan2(x,z) => angle around +Y axis where +Z is 0 rad
        return Mathf.Atan2(d.x, d.z);
    }

    private static bool TryBlockInterval(Vector3 shooterW, Vector3 blockerW, float radiusMeters, Interval goalCone, out Interval clipped)
    {
        clipped = default;

        Vector3 v = blockerW - shooterW;
        v.y = 0f;
        float d = v.magnitude;
        if (d < EPS) return false;

        // If blocker is behind shooter relative to goal cone center, it still could block side rays.
        // We'll rely on angular clipping to goalCone.

        // Angular half-width caused by radius at distance d:
        // delta = asin(r/d)
        float r = Mathf.Max(0f, radiusMeters);
        if (r <= 0f) return false;

        if (d <= r + 1e-3f)
        {
            // Essentially on shooter; blocks everything in cone
            clipped = new Interval(goalCone.a0, goalCone.a1);
            return true;
        }

        float delta = Mathf.Asin(Mathf.Clamp(r / d, 0f, 1f));
        float center = Mathf.Atan2(v.x, v.z);

        // Raw interval around center
        var raw = NormalizeInterval(center - delta, center + delta);

        // Map raw interval into goalCone unwrapped space
        // We want raw endpoints to be in same "turn" as goalCone
        float s = MapAngleToRange(raw.a0, goalCone.a0);
        float e = MapAngleToRange(raw.a1, goalCone.a0);

        // Ensure ordering
        if (e < s) e += 2f * Mathf.PI;

        // Clip to goalCone
        float cs = Mathf.Max(goalCone.a0, s);
        float ce = Mathf.Min(goalCone.a1, e);

        if (ce <= cs) return false;

        clipped = new Interval(cs, ce);
        return true;
    }

    private static float MapAngleToRange(float a, float baseA0)
    {
        // map 'a' to be >= baseA0 (possibly by adding 2pi)
        while (a < baseA0) a += 2f * Mathf.PI;
        while (a >= baseA0 + 2f * Mathf.PI) a -= 2f * Mathf.PI;
        return a;
    }

    // -----------------------------
    // Interval helpers (NO tuple fields)
    // -----------------------------
    private static float WrapPi(float a)
    {
        while (a <= -Mathf.PI) a += 2f * Mathf.PI;
        while (a > Mathf.PI) a -= 2f * Mathf.PI;
        return a;
    }

    private static Interval NormalizeInterval(float a0, float a1)
    {
        a0 = WrapPi(a0);
        a1 = WrapPi(a1);
        if (a1 < a0) a1 += 2f * Mathf.PI;
        return new Interval(a0, a1);
    }

    private static List<Interval> Merge(List<Interval> ints)
    {
        if (ints == null || ints.Count == 0) return ints ?? new List<Interval>();

        ints.Sort((p, q) => p.a0.CompareTo(q.a0));

        var res = new List<Interval>();
        for (int i = 0; i < ints.Count; i++)
        {
            var cur = ints[i];
            if (res.Count == 0) { res.Add(cur); continue; }

            var last = res[^1];

            if (cur.a0 <= last.a1)
            {
                last.a1 = Mathf.Max(last.a1, cur.a1);
                res[^1] = last;
            }
            else
            {
                res.Add(cur);
            }
        }

        return res;
    }

    private static List<Interval> ComplementIntervals(Interval goal, List<Interval> blocked)
    {
        var b = (blocked == null) ? new List<Interval>() : Merge(new List<Interval>(blocked));
        var open = new List<Interval>();

        float cur = goal.a0;

        for (int i = 0; i < b.Count; i++)
        {
            var bi = b[i];

            // clip into goal range
            float s = Mathf.Max(goal.a0, bi.a0);
            float e = Mathf.Min(goal.a1, bi.a1);
            if (e <= s) continue;

            if (s > cur)
                open.Add(new Interval(cur, s));

            cur = Mathf.Max(cur, e);
        }

        if (cur < goal.a1)
            open.Add(new Interval(cur, goal.a1));

        return open;
    }

    // -----------------------------
    // Mesh
    // -----------------------------
    private void EnsureMeshObjects()
    {
        if (_mf == null)
        {
            _mf = GetComponent<MeshFilter>();
            if (_mf == null) _mf = gameObject.AddComponent<MeshFilter>();
        }

        if (_mr == null)
        {
            _mr = GetComponent<MeshRenderer>();
            if (_mr == null) _mr = gameObject.AddComponent<MeshRenderer>();
        }

        if (openFillMaterial != null)
            _mr.sharedMaterial = openFillMaterial;
    }

    private void ClearMesh()
    {
        if (_mf != null) _mf.sharedMesh = null;
    }

    private void BuildOpenMesh(Computation comp)
    {
        EnsureMeshObjects();

        if (openFillMaterial != null)
            _mr.sharedMaterial = openFillMaterial;

        // Build a fan mesh from shooter, for each open interval:
        // shooter -> arc points at radius fillDistance -> back to shooter
        var verts = new List<Vector3>(1024);
        var tris = new List<int>(2048);

        Vector3 shooter = comp.shooterW;
        shooter.y = fillYOffset;

        for (int i = 0; i < comp.open.Count; i++)
        {
            var iv = comp.open[i];

            float span = Mathf.Max(0f, iv.a1 - iv.a0);
            if (span <= 1e-4f) continue;

            int segs = Mathf.Max(2, Mathf.RoundToInt(arcSegments * (span / (comp.goalCone.a1 - comp.goalCone.a0 + EPS))));
            int baseIndex = verts.Count;

            verts.Add(shooter); // center

            for (int s = 0; s <= segs; s++)
            {
                float t = (segs == 0) ? 0f : (s / (float)segs);
                float a = Mathf.Lerp(iv.a0, iv.a1, t);

                // point on XZ plane at distance fillDistance
                Vector3 dir = new Vector3(Mathf.Sin(a), 0f, Mathf.Cos(a));
                Vector3 p = shooter + dir * fillDistance;
                p.y = fillYOffset;
                verts.Add(p);
            }

            // triangles: (center, k, k+1)
            for (int k = 1; k <= segs; k++)
            {
                tris.Add(baseIndex);
                tris.Add(baseIndex + k);
                tris.Add(baseIndex + k + 1);
            }
        }

        var mesh = new Mesh();
        mesh.name = "OpenShotConeMesh";
        mesh.SetVertices(verts);
        mesh.SetTriangles(tris, 0);
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();

        _mf.sharedMesh = mesh;
    }

    // -----------------------------
    // Gizmos
    // -----------------------------
    private void OnDrawGizmos()
    {
        if (!drawGizmos) return;
        if (_last == null) return;
        if (mapper == null) return;

        // Ensure we have a computation
        if (_lastComp.blocked == null || _lastComp.open == null || _lastComp.blockers == null)
            _lastComp = Compute(_last);

        var comp = _lastComp;

        // shooter->posts lines
        Gizmos.color = shooterToPostColor;
        Gizmos.DrawLine(comp.shooterW, comp.goalLeftW);
        Gizmos.DrawLine(comp.shooterW, comp.goalRightW);

        // blocker circles (XZ)
        Gizmos.color = blockerCircleColor;
        foreach (var b in comp.blockers)
            DrawCircleXZ(b.posWorld, b.radiusMeters, 24);

        // draw arcs for blocked/open (pure debug)
        DrawIntervalArcs(comp, comp.blocked, blockedArcColor);
        DrawIntervalArcs(comp, comp.open, openArcColor);
    }

    private void DrawIntervalArcs(Computation comp, List<Interval> list, Color c)
    {
        if (list == null) return;

        Gizmos.color = c;

        Vector3 o = comp.shooterW;
        o.y = fillYOffset + 0.02f;

        float r = Mathf.Min(fillDistance, 30f);

        for (int i = 0; i < list.Count; i++)
        {
            var iv = list[i];
            int seg = 24;

            Vector3 prev = ArcPoint(o, r, iv.a0);
            for (int s = 1; s <= seg; s++)
            {
                float t = s / (float)seg;
                float a = Mathf.Lerp(iv.a0, iv.a1, t);
                Vector3 cur = ArcPoint(o, r, a);
                Gizmos.DrawLine(prev, cur);
                prev = cur;
            }
        }
    }

    private static Vector3 ArcPoint(Vector3 origin, float r, float a)
    {
        var p = origin + new Vector3(Mathf.Sin(a), 0f, Mathf.Cos(a)) * r;
        return p;
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
}