import os
import time

import mujoco
import mujoco.viewer
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

curr_dir = os.path.dirname(os.path.abspath(__file__))
mesh_dir = os.path.join(curr_dir, "RM75_6F")
scene_path = os.path.join(curr_dir, "sim_scene.xml")

assets = {}
if os.path.exists(mesh_dir):
    for file in os.listdir(mesh_dir):
        if file.lower().endswith(".stl"):
            with open(os.path.join(mesh_dir, file), 'rb') as f:
                key = f"package://rm_description/meshes/RM75_6F/{file}"
                assets[key] = f.read()
                print(f"Loaded asset: {file}")

try:
    model = mujoco.MjModel.from_xml_path(scene_path, assets=assets)
    data = mujoco.MjData(model)

    if model.nkey > 0:
        home_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_key_id != -1:
            data.qpos[:] = model.key_qpos[home_key_id]
            if model.nu > 0:
                data.ctrl[:] = model.key_ctrl[home_key_id]
            mujoco.mj_forward(model, data)

    print("\nScene loaded successfully! Launching Viewer...")

    camera_name = "d435_view"
    has_camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name) != -1

    if cv2 is None:
        print("OpenCV not found, running without camera window.")
        mujoco.viewer.launch(model, data)
    else:
        renderer = mujoco.Renderer(model, height=480, width=640)
        cv2.namedWindow(camera_name, cv2.WINDOW_NORMAL)

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()

                if has_camera:
                    renderer.update_scene(data, camera=camera_name)
                    image = renderer.render()
                    cv2.imshow(camera_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                else:
                    blank = 255 * np.ones((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        blank,
                        f"Camera '{camera_name}' not found",
                        (40, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )
                    cv2.imshow(camera_name, blank)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                time.sleep(0.01)

        renderer.close()
        cv2.destroyAllWindows()

except Exception as e:
    print(f"\nFailed to load scene!")
    print(f"Error detail: {e}")
