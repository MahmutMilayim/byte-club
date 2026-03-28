using UnityEngine;

public class TestAnimationInput : MonoBehaviour
{
    private PlayerAnimationDriver driver;

    void Awake()
    {
        driver = GetComponent<PlayerAnimationDriver>();
    }

    void Update()
    {
        if (Input.GetKey(KeyCode.W))
            driver.SetJog();
        else
            driver.SetIdle();

        if (Input.GetKeyDown(KeyCode.P))
            driver.PlayPass();

        if (Input.GetKeyDown(KeyCode.O))
            driver.PlayShoot();
    }
}