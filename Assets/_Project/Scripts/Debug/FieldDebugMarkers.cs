using UnityEngine;

public class FieldBoundsGizmo : MonoBehaviour
{
    public MetersToWorldMapper mapper;
    public float y = 0.1f;

    private void OnDrawGizmos()
    {
        if (mapper == null) return;

        // Köşeler (meters): (0,0), (68,0), (68,105), (0,105)
        Vector3 p00 = mapper.ToWorld(0f, 0f);    p00.y = y;
        Vector3 pW0 = mapper.ToWorld(68f, 0f);   pW0.y = y;
        Vector3 pWL = mapper.ToWorld(68f, 105f); pWL.y = y;
        Vector3 p0L = mapper.ToWorld(0f, 105f);  p0L.y = y;

        Gizmos.color = Color.yellow;
        Gizmos.DrawLine(p00, pW0);
        Gizmos.DrawLine(pW0, pWL);
        Gizmos.DrawLine(pWL, p0L);
        Gizmos.DrawLine(p0L, p00);
    }
}