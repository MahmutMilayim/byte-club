using UnityEngine;

public static class ShotOpennessEvaluator_SegmentCircle
{
    public static bool SegmentIntersectsCircle(Vector2 a, Vector2 b, Vector2 c, float r)
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
}