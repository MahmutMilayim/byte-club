using System.Collections.Generic;
using UnityEngine;

public class BestPassEvaluator : MonoBehaviour
{
    public FrameSnapshotApplier applier;
    public ShotGeometryConfig config;

    [Header("Pass rules")]
    public bool onlyOpponentsBlockPass = true;
    public float passBlockRadius = 0.85f;      // rakip oyuncu pası engellerken radius
    public float gkPassBlockRadius = 1.2f;     // kaleci pas lane'e giriyorsa

    [Header("Decision")]
    public float passMarginPercent = 5f;       // pass, shot'tan en az +5% iyiyse öner

    [Header("Log")]
    public bool logResult = true;

    private void OnEnable()
    {
        if (applier != null)
            applier.OnSnapshotApplied += _ => EvaluateNow();
    }

    private void OnDisable()
    {
        if (applier != null)
            applier.OnSnapshotApplied -= _ => EvaluateNow();
    }

    [ContextMenu("Evaluate Best Pass Now")]
    public void EvaluateNow()
    {
        var dto = applier != null ? applier.LastDTO : null;
        if (dto == null || dto.shooter == null || config == null)
        {
            Debug.LogWarning("BestPassEvaluator: missing dto/shooter/config.");
            return;
        }

        string shooterTeam = dto.shooter.teamId;
        Vector2 S = new Vector2(dto.shooter.x, dto.shooter.y);

        // Shooter shot openness
        float shooterOpen = EvaluateShotAt(dto, S, shooterTeam);

        // Find best teammate pass
        int bestId = -1;
        float bestOpen = -1f;
        Vector2 bestPos = default;

        if (dto.players != null)
        {
            foreach (var p in dto.players)
            {
                if (p == null) continue;
                if (p.teamId != shooterTeam) continue; // teammate only

                Vector2 T = new Vector2(p.x, p.y);

                // pass clear?
                if (!IsPassClear(dto, S, T, shooterTeam))
                    continue;

                float open = EvaluateShotAt(dto, T, shooterTeam);

                if (open > bestOpen)
                {
                    bestOpen = open;
                    bestId = p.id;
                    bestPos = T;
                }
            }
        }

        if (!logResult) return;

        if (bestId == -1)
        {
            Debug.Log($"[PASS] frame={dto.frameIndex} shooterOpen={shooterOpen:P1} bestPass=NONE");
            return;
        }

        // Decision suggestion
        float shooterPct = shooterOpen * 100f;
        float bestPct = bestOpen * 100f;

        string decision = (bestPct >= shooterPct + passMarginPercent) ? "PASS" : "SHOT";

        Debug.Log(
            $"[PASS] frame={dto.frameIndex} shooterOpen={shooterPct:F1}% " +
            $"bestPassTo={bestId} bestPassOpen={bestPct:F1}% decision={decision}"
        );
    }

    private float EvaluateShotAt(FrameSnapshotDTO dto, Vector2 shooterPos, string shooterTeamId)
    {
        string targetGoal = (dto.targetGoal ?? "TOP").Trim().ToUpperInvariant();
        float goalY = (targetGoal == "BOTTOM") ? 0f : config.fieldLength;

        Vector2 leftPost  = new Vector2(config.goalCenterX - config.goalHalfWidth, goalY);
        Vector2 rightPost = new Vector2(config.goalCenterX + config.goalHalfWidth, goalY);

        var blockers = BuildShotBlockers(dto, shooterTeamId);

        var res = ShotOpennessEvaluator.Evaluate(shooterPos, leftPost, rightPost, blockers);
        return res.openRatio;
    }

    private List<ShotOpennessEvaluator.Blocker> BuildShotBlockers(FrameSnapshotDTO dto, string shooterTeamId)
    {
        var blockers = new List<ShotOpennessEvaluator.Blocker>(32);

        // GK blocks (if opponent)
        if (dto.goalkeeper != null && dto.goalkeeper.teamId != shooterTeamId)
        {
            blockers.Add(new ShotOpennessEvaluator.Blocker
            {
                pos = new Vector2(dto.goalkeeper.x, dto.goalkeeper.y),
                radius = config.gkBlockRadius
            });
        }

        // opponents block
        if (dto.players != null)
        {
            foreach (var p in dto.players)
            {
                if (p == null) continue;
                if (p.teamId == shooterTeamId) continue;

                blockers.Add(new ShotOpennessEvaluator.Blocker
                {
                    pos = new Vector2(p.x, p.y),
                    radius = config.playerBlockRadius
                });
            }
        }

        return blockers;
    }

    private bool IsPassClear(FrameSnapshotDTO dto, Vector2 S, Vector2 T, string shooterTeamId)
    {
        // Opponents block the pass
        if (dto.players != null)
        {
            foreach (var p in dto.players)
            {
                if (p == null) continue;

                if (onlyOpponentsBlockPass && p.teamId == shooterTeamId)
                    continue;

                // Teammate receiver itself shouldn't be treated as blocker if close
                if ((new Vector2(p.x, p.y) - T).sqrMagnitude < 0.0001f) continue;

                float r = passBlockRadius;

                if (ShotOpennessEvaluator_SegmentCircle.SegmentIntersectsCircle(S, T, new Vector2(p.x, p.y), r))
                    return false;
            }
        }

        // GK can also block a pass lane (optional)
        if (dto.goalkeeper != null)
        {
            if (!onlyOpponentsBlockPass || dto.goalkeeper.teamId != shooterTeamId)
            {
                float r = gkPassBlockRadius;
                if (ShotOpennessEvaluator_SegmentCircle.SegmentIntersectsCircle(S, T, new Vector2(dto.goalkeeper.x, dto.goalkeeper.y), r))
                    return false;
            }
        }

        return true;
    }
}