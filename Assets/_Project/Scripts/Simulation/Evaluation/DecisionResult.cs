using System;
using System.Collections.Generic;
using UnityEngine;

[Serializable]
public class PassOptionResult
{
    public int receiverId;
    public bool passClear;
    public float shotOpenRatio;   // 0..1 ; pass clear değilse -1 olabilir
    public Vector2 receiverMeters;
}

[Serializable]
public class DecisionResult
{
    public int frameIndex;

    public int initialShooterId;
    public float initialShooterOpenRatio;   // 0..1

    public bool usePass;
    public int passReceiverId;              // usePass false ise -1
    public Vector2 passReceiverMeters;

    public int finalShooterId;
    public float finalShooterOpenRatio;     // 0..1

    public List<PassOptionResult> passOptions = new();
}