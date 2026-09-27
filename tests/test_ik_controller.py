from time import time

import gymnasium as gym
import numpy as np
import pytest
import torch

import mani_skill.envs
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common


def test_fetch_pd_ee_delta_pose_named_root_cpu():
    env = gym.make(
        "PushCube-v1",
        robot_uids="fetch",
        control_mode="pd_ee_delta_pose",
        num_envs=1,
        obs_mode="state",
        reward_mode="none",
        sim_backend="physx_cpu",
        render_backend="none",
        robot_init_qpos_noise=0.0,
    )
    try:
        env.reset(seed=0)
        arm = env.unwrapped.agent.controller.controllers["arm"]
        assert arm.config.root_link_name == "torso_lift_link"
        assert arm.root_link.name == "torso_lift_link"
        torch.testing.assert_close(
            arm.ee_pose_at_base.raw_pose,
            (arm.root_link.pose.inv() * arm.ee_link.pose).raw_pose,
        )
    finally:
        env.close()


@pytest.mark.gpu_sim
@pytest.mark.parametrize("control_mode", ["pd_ee_delta_pose", "pd_ee_target_delta_pose", "pd_ee_delta_pos", "pd_ee_target_delta_pos"])
def test_pd_ee_delta_controller(control_mode):
    gpu_env = gym.make(
        "PickCube-v1",
        num_envs=16,
        obs_mode="state",
        control_mode=control_mode,
        sim_backend="physx_cuda",
        robot_init_qpos_noise=0.0,
    )
    env = gym.make(
        "PickCube-v1",
        num_envs=1,
        obs_mode="state",
        control_mode=control_mode,
        sim_backend="physx_cpu",
        robot_init_qpos_noise=0.0,
    )
    cpu_base_env: BaseEnv = env.unwrapped
    gpu_base_env: BaseEnv = gpu_env.unwrapped
    
    env.reset(seed=0)
    gpu_env.reset(seed=0)
    action = common.to_tensor(env.action_space.sample())
    action[:3] = 0.02
    if "pose" in control_mode:
        action[3:] = 0.1
    for i in range(5):
        env.step(action)
        gpu_env.step(action.expand(gpu_base_env.num_envs, -1))

    ee_link_name = cpu_base_env.agent.controller.controllers["arm"].config.ee_link
    link = cpu_base_env.agent.robot.links_map[ee_link_name]
    gpu_link = gpu_base_env.agent.robot.links_map[ee_link_name]
    np.testing.assert_allclose(
        common.to_numpy(link.pose.p.mean(0)), common.to_numpy(gpu_link.pose.p.mean(0)), atol=5e-4
    )
    np.testing.assert_allclose(
        common.to_numpy(link.pose.q.mean(0)), common.to_numpy(gpu_link.pose.q.mean(0)), atol=5e-4
    )
    env.close()
    gpu_env.close()

@pytest.mark.gpu_sim
@pytest.mark.parametrize("control_mode", ["pd_ee_pose"])
def test_pd_ee_controller(control_mode):
    gpu_env = gym.make(
        "PickCube-v1",
        num_envs=16,
        obs_mode="state",
        control_mode=control_mode,
        sim_backend="physx_cuda",
    )
    env = gym.make(
        "PickCube-v1",
        num_envs=1,
        obs_mode="state",
        control_mode=control_mode,
        sim_backend="physx_cpu",
    )
    cpu_base_env: BaseEnv = env.unwrapped
    gpu_base_env: BaseEnv = gpu_env.unwrapped

    env.reset(seed=0)
    gpu_env.reset(seed=0)
    target_pose = torch.tensor([0.4, 0.1, 0.5, np.pi / 2, 0.0, 0.0, 10.0])
    # we take a lot of extra steps here since with the absolute pd ee pose control, we want to converge to the target pose.
    # this will often take multiple steps. Moreover by default the IK solver for the GPU sim is a single iteration only.
    for i in range(20):
        env.step(target_pose)
        gpu_env.step(target_pose.expand(gpu_base_env.num_envs, -1))

    ee_link_name = cpu_base_env.agent.controller.controllers["arm"].config.ee_link
    link = cpu_base_env.agent.robot.links_map[ee_link_name]
    gpu_link = gpu_base_env.agent.robot.links_map[ee_link_name]
    np.testing.assert_allclose(
        common.to_numpy(link.pose.p.mean(0)), common.to_numpy(gpu_link.pose.p.mean(0)), atol=5e-4
    )
    np.testing.assert_allclose(
        common.to_numpy(link.pose.q.mean(0)), common.to_numpy(gpu_link.pose.q.mean(0)), atol=5e-4
    )
    env.close()
    gpu_env.close()
