using System;

[Serializable]
public class ShotScoreEntry
{
    public int playerId;
    public bool isInitialShooter;
    public bool passClear;
    public float scorePercent;
}