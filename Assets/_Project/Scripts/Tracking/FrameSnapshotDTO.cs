using System;
using UnityEngine;

[Serializable]
public class FrameSnapshotDTO
{
    public int frameIndex;

    // ✅ EKLENDİ: JSON'daki shooter objesini buraya koyacağız
    public ShooterDTO shooter;

    // ✅ Opsiyonel: JSON'da goal alanı varsa okur
    public GoalDTO goal;

    public PlayerDTO[] players;
    public BallDTO ball;

    [Serializable]
    public class PlayerDTO
    {
        public int id;
        public string teamId; // ✅ İstersen JSON'da teamId de gelsin
        public float x;       // meters: 0..68
        public float y;       // meters: 0..105
    }

    [Serializable]
    public class ShooterDTO
    {
        public int playerId;
        public string teamId;
        public string targetGoal; // "TOP" or "BOTTOM"
    }

    [Serializable]
    public class GoalDTO
    {
        public string target; // "TOP" or "BOTTOM"
    }

    [Serializable]
    public class BallDTO
    {
        public bool visible;
        public float x; // meters
        public float y; // meters
    }
}