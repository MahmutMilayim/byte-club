using UnityEngine;

[CreateAssetMenu(menuName="Football/Shot Geometry Config", fileName="ShotGeometryConfig")]
public class ShotGeometryConfig : ScriptableObject
{
    public float fieldWidth = 68f;
    public float fieldLength = 105f;

    public float goalWidth = 7.32f;          // FIFA
    public float playerBlockRadius = 0.6f;   // tuning (omuz+pay)
    public float gkBlockRadius = 1.1f;       // tuning (daha büyük)

    public float goalCenterX => fieldWidth * 0.5f; // 34
    public float goalHalfWidth => goalWidth * 0.5f; // 3.66
}