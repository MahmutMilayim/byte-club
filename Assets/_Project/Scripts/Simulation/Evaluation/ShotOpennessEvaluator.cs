using System;
using System.Collections.Generic;
using UnityEngine;

public static class ShotOpennessEvaluator
{
    public struct Blocker
    {
        public Vector2 pos;   // meters (x,y)
        public float radius;  // meters
    }

    public struct Result
    {
        public float goalAngle;     // radians
        public float blockedAngle;  // radians
        public float openRatio;     // 0..1
        public List<(float a0, float a1)> blockedIntervals; // normalized
    }

    public static Result Evaluate(
        Vector2 shooter,
        Vector2 leftPost,
        Vector2 rightPost,
        IReadOnlyList<Blocker> blockers)
    {
        // 1) Kale açı aralığı
        float aL = Angle(shooter, leftPost);
        float aR = Angle(shooter, rightPost);

        // -pi..pi wrap sorununu çöz: interval’i normalize et
        var goal = NormalizeInterval(aL, aR); // [a0,a1] where a0<=a1 in "unwrapped" space
        float goalAngle = goal.a1 - goal.a0;

        if (goalAngle <= 1e-5f)
        {
            return new Result
            {
                goalAngle = 0f,
                blockedAngle = 0f,
                openRatio = 0f,
                blockedIntervals = new List<(float, float)>()
            };
        }

        // 2) Blocker interval’leri: teğet açı aralığı
        var clipped = new List<(float a0, float a1)>();

        foreach (var b in blockers)
        {
            if (!TryBlockerInterval(shooter, b.pos, b.radius, out var bi))
                continue;

            // Goal interval ile kesiştir
            if (Intersect(goal, bi, out var inter))
                clipped.Add(inter);
        }

        // 3) Union
        var merged = MergeIntervals(clipped);

        float blocked = 0f;
        foreach (var m in merged)
            blocked += (m.a1 - m.a0);

        float openRatio = Mathf.Clamp01(1f - blocked / goalAngle);

        return new Result
        {
            goalAngle = goalAngle,
            blockedAngle = blocked,
            openRatio = openRatio,
            blockedIntervals = merged
        };
    }

    private static float Angle(Vector2 from, Vector2 to)
    {
        Vector2 d = to - from;
        return Mathf.Atan2(d.y, d.x); // radians
    }

    private static bool TryBlockerInterval(Vector2 shooter, Vector2 c, float r, out (float a0, float a1) interval)
    {
        Vector2 v = c - shooter;
        float d = v.magnitude;

        // shooter ile aynı nokta / çok yakınsa
        if (d <= 1e-5f)
        {
            interval = default;
            return false;
        }

        // Disk shooter'ı içeriyorsa: tüm açıları kapatıyor gibi davran
        if (d <= r)
        {
            // devasa interval: goal ile intersect edilince full kapatır
            float a = Mathf.Atan2(v.y, v.x);
            interval = NormalizeInterval(a - Mathf.PI, a + Mathf.PI);
            return true;
        }

        float alpha = Mathf.Atan2(v.y, v.x);
        float delta = Mathf.Asin(Mathf.Clamp(r / d, -1f, 1f));

        float a0 = alpha - delta;
        float a1 = alpha + delta;

        interval = NormalizeInterval(a0, a1);
        return true;
    }

    // ---- Interval helpers (unwrap) ----
    private static (float a0, float a1) NormalizeInterval(float a0, float a1)
    {
        // hedef: a1 >= a0 olacak şekilde unwrap et
        // wrap farkı > PI ise, daha kısa yoldan değil "aynı yönde" aç
        a0 = WrapPi(a0);
        a1 = WrapPi(a1);

        if (a1 < a0) a1 += 2f * Mathf.PI; // unwrap

        // bazen a1-a0 çok büyük olabilir (disk contains shooter) -> bırak
        return (a0, a1);
    }

    private static float WrapPi(float a)
    {
        while (a <= -Mathf.PI) a += 2f * Mathf.PI;
        while (a > Mathf.PI) a -= 2f * Mathf.PI;
        return a;
    }

    private static bool Intersect((float a0, float a1) A, (float a0, float a1) B, out (float a0, float a1) I)
    {
        // B’yi A’ya yakın unwrap etmek için gerekirse 2π ekle/çıkar
        B = AlignTo(A, B);

        float s = Mathf.Max(A.a0, B.a0);
        float e = Mathf.Min(A.a1, B.a1);

        if (e > s)
        {
            I = (s, e);
            return true;
        }

        I = default;
        return false;
    }

    private static (float a0, float a1) AlignTo((float a0, float a1) refI, (float a0, float a1) I)
    {
        // I’yi ref aralığının yakınında tut (2π kaydır)
        float k = Mathf.Round((refI.a0 - I.a0) / (2f * Mathf.PI));
        float shift = k * 2f * Mathf.PI;
        return (I.a0 + shift, I.a1 + shift);
    }

    private static List<(float a0, float a1)> MergeIntervals(List<(float a0, float a1)> ints)
    {
        ints.Sort((p, q) => p.a0.CompareTo(q.a0));
        var res = new List<(float a0, float a1)>();

        foreach (var cur in ints)
        {
            if (res.Count == 0) { res.Add(cur); continue; }

            var last = res[res.Count - 1];
            if (cur.a0 <= last.a1)
            {
                last.a1 = Mathf.Max(last.a1, cur.a1);
                res[res.Count - 1] = last;
            }
            else res.Add(cur);
        }
        return res;
    }
}