import os
import sys
import argparse
import numpy as np

from isaacgym import gymapi, gymutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run without viewer")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device id")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of envs")
    args = parser.parse_args()

    # ---------------------------
    # Resolve project root (assumes this file is in <repo>/scripts/)
    # ---------------------------
    this_file = os.path.abspath(__file__)
    repo_root = os.path.dirname(os.path.dirname(this_file))  # <repo>
    urdf_rel_path = "data/data_urdf/robot/wujihand/wujihand_right_isaac_fixed.urdf"


    asset_root = repo_root
    asset_file = urdf_rel_path

    urdf_abs = os.path.join(asset_root, asset_file)
    if not os.path.exists(urdf_abs):
        raise FileNotFoundError(f"URDF not found: {urdf_abs}")

    # ---------------------------
    # Isaac Gym init
    # ---------------------------
    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2

    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

    # PhysX settings (typical stable defaults)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = 8
    sim_params.physx.use_gpu = True

    compute_device_id = args.gpu
    graphics_device_id = -1 if args.headless else args.gpu

    sim = gym.create_sim(compute_device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create sim")

    # Ground plane (optional, harmless even if hand is fixed)
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    # ---------------------------
    # Load WujiHand URDF asset
    # ---------------------------
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.disable_gravity = False
    asset_options.flip_visual_attachments = False
    asset_options.use_mesh_materials = True
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    if asset is None:
        raise RuntimeError(f"Failed to load asset: root={asset_root}, file={asset_file}")

    num_dofs = gym.get_asset_dof_count(asset)
    dof_props = gym.get_asset_dof_properties(asset)

    print("\n[WujiHand] Loaded asset:")
    print("  asset_root:", asset_root)
    print("  asset_file:", asset_file)
    print("  num_dofs:", num_dofs)

    # Print DOF info for mapping/debug
    for i in range(num_dofs):
        name = gym.get_asset_dof_name(asset, i)
        lo = float(dof_props["lower"][i])
        hi = float(dof_props["upper"][i])
        print(f"  DOF[{i:02d}] {name:40s}  [{lo:.4f}, {hi:.4f}]")

    # Safe default pose: mid-range
    lower = dof_props["lower"].copy()
    upper = dof_props["upper"].copy()
    default_dof_pos = 0.5 * (lower + upper)

    # PD gains (start conservative)
    dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
    dof_props["stiffness"].fill(200.0)
    dof_props["damping"].fill(20.0)

    # ---------------------------
    # Create env(s) and actor(s)
    # ---------------------------
    spacing = 1.0
    env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)

    num_envs = args.num_envs
    envs = []
    actors = []

    # Place hand above ground a bit
    hand_pose = gymapi.Transform()
    hand_pose.p = gymapi.Vec3(0.0, 0.0, 0.5)
    hand_pose.r = gymapi.Quat(0, 0, 0, 1)

    for env_id in range(num_envs):
        env = gym.create_env(sim, env_lower, env_upper, int(np.ceil(np.sqrt(num_envs))))
        envs.append(env)

        actor = gym.create_actor(env, asset, hand_pose, f"wujihand_{env_id}", env_id, 1)
        actors.append(actor)

        # Apply DOF properties + initial state
        gym.set_actor_dof_properties(env, actor, dof_props)

        dof_state = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
        dof_state["pos"] = default_dof_pos
        dof_state["vel"] = 0.0
        gym.set_actor_dof_states(env, actor, dof_state, gymapi.STATE_ALL)

        # Set position targets to hold pose (important for DOF_MODE_POS)
        gym.set_actor_dof_position_targets(env, actor, default_dof_pos)

    # ---------------------------
    # Viewer
    # ---------------------------
    viewer = None
    if not args.headless:
        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise RuntimeError("Failed to create viewer")

        cam_pos = gymapi.Vec3(1.0, 1.0, 1.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
        gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # ---------------------------
    # Main loop
    # ---------------------------
    t = 0.0
    while True:
        if viewer is not None and gym.query_viewer_has_closed(viewer):
            break

        # Example: small sinusoidal motion on first DOF (if exists)
        if num_dofs > 0:
            target = default_dof_pos.copy()
            target[0] = default_dof_pos[0] + 0.2 * np.sin(2.0 * np.pi * 0.5 * t)
            for env, actor in zip(envs, actors):
                gym.set_actor_dof_position_targets(env, actor, target)

        gym.simulate(sim)
        gym.fetch_results(sim, True)

        if viewer is not None:
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)

        gym.sync_frame_time(sim)
        t += sim_params.dt

    # Cleanup
    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
