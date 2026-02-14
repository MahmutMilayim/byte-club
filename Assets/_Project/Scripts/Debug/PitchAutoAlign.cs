using UnityEngine;

[ExecuteAlways]
public class PitchAutoAlign : MonoBehaviour
{
    [Header("Refs")]
    public MetersToWorldMapper mapper;
    public Transform pitchPlane; // sahadaki plane object (MeshRenderer olan)

    [Header("Options")]
    public bool setPosition = true;
    public bool setRotation = true;
    public bool setScale = true;

    [Tooltip("Unity built-in Plane mesh size is 10x10 units")]
    public float unityPlaneSize = 10f;

    [ContextMenu("Align Now")]
    public void AlignNow()
    {
        if (mapper == null || pitchPlane == null)
        {
            Debug.LogError("PitchAutoAlign: mapper or pitchPlane missing.");
            return;
        }

        // Meter-space field center in WORLD:
        // meters center = (fieldWidth/2, fieldLength/2)
        // We convert to world, then force pitchPlane to match.
        Vector3 worldCenter = mapper.ToWorld(mapper.fieldWidth * 0.5f, mapper.fieldLength * 0.5f);
        worldCenter.y = 0f;

        if (setPosition)
            pitchPlane.position = worldCenter;

        if (setRotation)
            pitchPlane.rotation = Quaternion.Euler(0f, 0f, 0f);

        if (setScale)
        {
            // Plane local X corresponds to world X, Plane local Z corresponds to world Z.
            // We want Plane world size to match mapper's field world extents.
            float worldSizeX = mapper.fieldLength; // because lengthAlongX = true -> world X spans 105
            float worldSizeZ = mapper.fieldWidth;  // world Z spans 68

            // If mapper lengthAlongX is false, swap
            // (We infer it safely by sampling 2 points)
            Vector3 p0 = mapper.ToWorld(0f, 0f);
            Vector3 pL = mapper.ToWorld(0f, mapper.fieldLength);
            Vector3 pW = mapper.ToWorld(mapper.fieldWidth, 0f);

            float spanAlongX_fromLength = Mathf.Abs(pL.x - p0.x);
            float spanAlongZ_fromLength = Mathf.Abs(pL.z - p0.z);

            bool lengthAlongX = spanAlongX_fromLength > spanAlongZ_fromLength;

            if (!lengthAlongX)
            {
                worldSizeX = mapper.fieldWidth;
                worldSizeZ = mapper.fieldLength;
            }

            pitchPlane.localScale = new Vector3(worldSizeX / unityPlaneSize, 1f, worldSizeZ / unityPlaneSize);
        }
    }

    private void OnValidate()
    {
        // Inspector değişince otomatik hizalasın
        AlignNow();
    }
}