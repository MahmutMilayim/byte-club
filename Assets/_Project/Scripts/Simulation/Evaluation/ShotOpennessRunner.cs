using System.Collections.Generic;
using UnityEngine;

public class ShotOpennessRunner : MonoBehaviour
{
    public FrameSnapshotApplier applier;
    public ShotGeometryConfig config;

    [Header("Debug")]
    public bool logResult = true;
    public bool onlyOpponentsBlock = true;

    // ✅ Visualization için dışarı açıyoruz
    public ShotOpennessEvaluator.Result LastResult { get; private set; }
    public Vector2 LastShooterMeters { get; private set; }
    public Vector2 LastLeftPostMeters { get; private set; }
    public Vector2 LastRightPostMeters { get; private set; }
    public string LastTargetGoal { get; private set; } = "TOP";
    public string LastShooterTeamId { get; private set; } = "";

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

    [ContextMenu("Evaluate Now")]
    public void EvaluateNow()
    {
        var dto = applier != null ? applier.LastDTO : null;
        if (dto == null || dto.shooter == null || config == null)
        {
            Debug.LogWarning("ShotOpennessRunner: missing dto/shooter/config.");
            return;
        }

        LastShooterTeamId = dto.shooter.teamId ?? "";
        LastShooterMeters = new Vector2(dto.shooter.x, dto.shooter.y);

        LastTargetGoal = (dto.targetGoal ?? "TOP").Trim().ToUpperInvariant();
        float goalY = (LastTargetGoal == "BOTTOM") ? 0f : config.fieldLength;

        LastLeftPostMeters  = new Vector2(config.goalCenterX - config.goalHalfWidth, goalY);
        LastRightPostMeters = new Vector2(config.goalCenterX + config.goalHalfWidth, goalY);

        var blockers = new List<ShotOpennessEvaluator.Blocker>(32);

        // GK (rakipse)
        if (dto.goalkeeper != null && (!onlyOpponentsBlock || dto.goalkeeper.teamId != LastShooterTeamId))
        {
            blockers.Add(new ShotOpennessEvaluator.Blocker
            {
                pos = new Vector2(dto.goalkeeper.x, dto.goalkeeper.y),
                radius = config.gkBlockRadius
            });
        }

        // Players (rakipler)
        if (dto.players != null)
        {
            foreach (var p in dto.players)
            {
                if (p == null) continue;
                if (onlyOpponentsBlock && p.teamId == LastShooterTeamId) continue;

                blockers.Add(new ShotOpennessEvaluator.Blocker
                {
                    pos = new Vector2(p.x, p.y),
                    radius = config.playerBlockRadius
                });
            }
        }

        LastResult = ShotOpennessEvaluator.Evaluate(LastShooterMeters, LastLeftPostMeters, LastRightPostMeters, blockers);

        if (logResult)
            Debug.Log($"Shot openness: {LastResult.openRatio:P1} | blockedAngle={LastResult.blockedAngle:F3} / goalAngle={LastResult.goalAngle:F3}");
    }
}