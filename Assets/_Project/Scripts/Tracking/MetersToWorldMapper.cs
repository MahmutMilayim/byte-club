using UnityEngine;

[CreateAssetMenu(menuName = "Football/Meters To World Mapper", fileName = "MetersToWorldMapper")]
public class MetersToWorldMapper : ScriptableObject
{
    [Header("Field size (meters)")]
    public float fieldWidth = 68f;   // X
    public float fieldLength = 105f; // Z

    [Header("Placement")]
    public bool originAtCenter = true;
    public float yOffset = 0.05f; // z-fighting için

    public Vector3 ToWorld(float metersX, float metersY)
    {
        // metersX: 0..68, metersY: 0..105
        float worldX = metersX;
        float worldZ = metersY;

        if (originAtCenter)
        {
            worldX -= fieldWidth * 0.5f;   // -34..+34
            worldZ -= fieldLength * 0.5f;  // -52.5..+52.5
        }

        return new Vector3(worldX, yOffset, worldZ);
    }
}