using UnityEngine;

[CreateAssetMenu(menuName = "Football/Meters To World Mapper", fileName = "MetersToWorldMapper")]
public class MetersToWorldMapper : ScriptableObject
{
    [Header("Field size (meters)")]
    public float fieldWidth = 68f;    // width
    public float fieldLength = 105f;  // length

    [Header("Placement")]
    public bool originAtCenter = true;
    public float yOffset = 0.05f;

    [Header("Axis Mapping")]
    [Tooltip("True ise saha uzunluğu (105m) world X eksenine yerleşir. Kaleler sağ-sol olur.")]
    public bool lengthAlongX = true; // ✅ senin durumunda TRUE olmalı

    [Header("Data Fix (optional)")]
    public bool swapXY = false;
    public bool flipLength = false; // uzunluk eksenini ters çevir (kale yönü)

    public Vector3 ToWorld(float metersX, float metersY)
    {
        // Beklenen: metersX: 0..68 (width), metersY: 0..105 (length)

        if (swapXY)
            (metersX, metersY) = (metersY, metersX);

        if (flipLength)
            metersY = fieldLength - metersY;

        float worldX, worldZ;

        if (lengthAlongX)
        {
            // ✅ length -> X, width -> Z
            worldX = metersY;  // 0..105
            worldZ = metersX;  // 0..68
        }
        else
        {
            // length -> Z, width -> X (eski davranış)
            worldX = metersX;  // 0..68
            worldZ = metersY;  // 0..105
        }

        if (originAtCenter)
        {
            if (lengthAlongX)
            {
                worldX -= fieldLength * 0.5f; // -52.5..+52.5
                worldZ -= fieldWidth * 0.5f;  // -34..+34
            }
            else
            {
                worldX -= fieldWidth * 0.5f;  // -34..+34
                worldZ -= fieldLength * 0.5f; // -52.5..+52.5
            }
        }

        return new Vector3(worldX, yOffset, worldZ);
    }
}