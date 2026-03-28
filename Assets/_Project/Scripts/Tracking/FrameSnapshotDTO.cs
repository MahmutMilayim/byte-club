using System;
using UnityEngine;

[Serializable]
public class FrameSnapshotDTO
{
    public int frameIndex;

    // Hangi kaleye şut atılıyor (frame-level)
    public string targetGoal; // "TOP" or "BOTTOM"

    // Şut atan oyuncu (pozisyon dahil)
    public ShooterDTO shooter;

    // Rakip kaleci (pozisyon dahil)
    public GoalkeeperDTO goalkeeper;

    // Diğer tüm oyuncular (role yok)
    public PlayerDTO[] players;

    public BallDTO ball;

    // -------------------------
    // Nested DTO Classes
    // -------------------------

    [Serializable]
    public class ShooterDTO
    {
        public int playerId;
        public string teamId;
        public float x;  // meters: 0..68
        public float y;  // meters: 0..105
    }

    [Serializable]
    public class GoalkeeperDTO
    {
        public int playerId;
        public string teamId;
        public float x;
        public float y;
    }

    [Serializable]
    public class PlayerDTO
    {
        public int id;
        public string teamId;
        public float x;
        public float y;
    }

    [Serializable]
    public class BallDTO
    {
        public bool visible;
        public float x;
        public float y;
    }
}