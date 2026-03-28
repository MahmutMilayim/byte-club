using UnityEngine;

public class ShotDebugDrawer : MonoBehaviour
{
    public FrameSnapshotApplier applier;
    public ShotGeometryConfig config;

    [Header("Draw")]
    public bool draw = true;
    public float shooterSize = 0.3f;
    public float postSize = 0.25f;

    private void OnDrawGizmos()
    {
        if (!draw || applier == null || config == null) return;

        var dto = applier.LastDTO;
        if (dto == null || dto.shooter == null) return;

        Vector2 S = new Vector2(dto.shooter.x, dto.shooter.y);

        string targetGoal = (dto.targetGoal ?? "TOP").Trim().ToUpperInvariant();
        float goalY = (targetGoal == "BOTTOM") ? 0f : config.fieldLength;

        Vector2 L = new Vector2(config.goalCenterX - config.goalHalfWidth, goalY);
        Vector2 R = new Vector2(config.goalCenterX + config.goalHalfWidth, goalY);

        // Shooter
        Gizmos.color = Color.yellow;
        Gizmos.DrawSphere(new Vector3(S.x, 0.2f, S.y), shooterSize);

        // Posts
        Gizmos.color = Color.white;
        Gizmos.DrawSphere(new Vector3(L.x, 0.2f, L.y), postSize);
        Gizmos.DrawSphere(new Vector3(R.x, 0.2f, R.y), postSize);

        // Rays
        Gizmos.color = Color.cyan;
        Gizmos.DrawLine(new Vector3(S.x, 0.2f, S.y), new Vector3(L.x, 0.2f, L.y));
        Gizmos.DrawLine(new Vector3(S.x, 0.2f, S.y), new Vector3(R.x, 0.2f, R.y));

        // GK
        if (dto.goalkeeper != null)
        {
            Gizmos.color = Color.red;
            Vector2 gk = new Vector2(dto.goalkeeper.x, dto.goalkeeper.y);
            Gizmos.DrawWireSphere(new Vector3(gk.x, 0.2f, gk.y), config.gkBlockRadius);
        }

        // Opponents
        if (dto.players != null)
        {
            foreach (var p in dto.players)
            {
                if (p == null) continue;
                if (p.teamId == dto.shooter.teamId) continue;

                Gizmos.color = Color.magenta;
                Vector2 c = new Vector2(p.x, p.y);
                Gizmos.DrawWireSphere(new Vector3(c.x, 0.2f, c.y), config.playerBlockRadius);
            }
        }
    }
}