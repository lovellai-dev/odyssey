"""Tests for the UR5e drug-sorting embodiment config assembly (modality/info)."""

from __future__ import annotations

from odyssey.embodiments.ur5e_drugsort import embodiment as emb


def test_dims_are_consistent() -> None:
    assert emb.ARM_DIM == 6
    assert emb.STATE_DIM == 7
    assert emb.ACTION_DIM == 7
    assert len(emb.FEATURE_NAMES) == emb.STATE_DIM
    assert emb.FEATURE_NAMES[-1] == "gripper.pos"


def test_modality_json_slices_cover_the_vector() -> None:
    m = emb.modality_json()
    for group in ("state", "action"):
        assert m[group]["single_arm"] == {"start": 0, "end": 6}
        assert m[group]["gripper"] == {"start": 6, "end": 7}
        # slices tile [0, STATE_DIM) with no gap/overlap
        assert m[group]["single_arm"]["end"] == m[group]["gripper"]["start"]
        assert m[group]["gripper"]["end"] == emb.STATE_DIM


def test_modality_video_keys_map_to_observation_images() -> None:
    m = emb.modality_json()
    # Two views = the Playground camera-mounts published each tick: the fixed
    # `exterior` overview and the `wrist` eye-in-hand (iteration 3 grasp depth).
    assert set(m["video"]) == {"exterior", "wrist"}
    assert m["video"]["exterior"]["original_key"] == "observation.images.exterior"
    assert m["video"]["wrist"]["original_key"] == "observation.images.wrist"
    assert m["annotation"]["human.task_description"]["original_key"] == "task_index"


def test_features_shapes_and_names() -> None:
    feats = emb.features(width=256, height=256, fps=20)
    assert feats["action"]["shape"] == [7]
    assert feats["observation.state"]["shape"] == [7]
    assert feats["action"]["names"] == list(emb.FEATURE_NAMES)
    for key in ("observation.images.exterior", "observation.images.wrist"):
        assert feats[key]["dtype"] == "video"
        assert feats[key]["shape"] == [256, 256, 3]
        assert feats[key]["info"]["video.fps"] == 20
        assert feats[key]["info"]["video.codec"] == "h264"


def test_info_json_top_level_fields() -> None:
    info = emb.info_json(
        total_episodes=3, total_frames=1500, total_tasks=1, fps=20, width=256, height=256
    )
    assert info["codebase_version"] == "v2.1"
    assert info["robot_type"] == "ur5e_robotiq_2f85"
    assert info["total_episodes"] == 3
    assert info["total_frames"] == 1500
    assert info["total_videos"] == 6  # 3 episodes x 2 cameras (exterior + wrist)
    assert info["total_chunks"] == 1
    assert info["fps"] == 20
    assert info["splits"] == {"train": "0:3"}
    assert "episode_{episode_index:06d}.parquet" in info["data_path"]
    assert set(info["features"]) >= {
        "action", "observation.state",
        "observation.images.exterior", "observation.images.wrist",
        "timestamp", "frame_index", "episode_index", "index", "task_index",
    }
