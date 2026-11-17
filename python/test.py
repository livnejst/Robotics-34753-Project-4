import time
from MyRobot import MyRobot

PERIOD = 5.0          # seconds between readings
TORQUE_OFF = True   # True -> release the motors so you can move the arm by hand

robot = MyRobot()     # moves to (0, 0, 0, 0)
if robot.init_status:
    try:
        if TORQUE_OFF:
            input("Hold the arm and press Enter to release torque...")
            for dxl_id in robot.motor_ids:
                robot._write1(dxl_id, robot.ADDR_MX_TORQUE_ENABLE, robot.TORQUE_DISABLE)

        print("Reading joint angles (Ctrl+C to stop)...\n")
        while True:
            try:
                q = robot.read_joint_angles()
            except IOError as e:
                print(f"Read error: {e}")
                time.sleep(PERIOD)
                continue

            print("  ".join(f"Joint {i}: {a:7.2f}°" for i, a in enumerate(q, start=1)))
            time.sleep(PERIOD)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if not TORQUE_OFF:
            input("Hold the arm and press Enter to release torque...")
        robot.disable_motors()
