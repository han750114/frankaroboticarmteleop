import pyrealsense2 as rs

ctx = rs.context()
if len(ctx.devices) == 0:
    print("❌ No RealSense devices connected.")
else:
    for dev in ctx.devices:
        print("✅ Found device:")
        print("  Name:", dev.get_info(rs.camera_info.name))
        print("  Serial:", dev.get_info(rs.camera_info.serial_number))
