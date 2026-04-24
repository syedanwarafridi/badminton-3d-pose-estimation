import numpy as np
import json
import time
import os


class PoseEstimator3D:
    def __init__(self):
        self.previous_poses = {}

        self.body_height_map = {
            0: 1.70, 1: 1.68, 2: 1.68, 3: 1.66, 4: 1.66,
            5: 1.45, 6: 1.45, 7: 1.20, 8: 1.20, 9: 1.05, 10: 1.05,
            11: 0.95, 12: 0.95, 13: 0.50, 14: 0.50, 15: 0.04, 16: 0.04
        }

        self.scale_factor = 0.003

    def estimate_pose(self, keypoints_2d, ankle_world):
        pose_3d = np.zeros((17, 3))

        pose_3d[15] = [ankle_world[0][0], ankle_world[0][1], 0.04]
        pose_3d[16] = [ankle_world[1][0], ankle_world[1][1], 0.04]

        ankle_center_world = (pose_3d[15, :2] + pose_3d[16, :2]) / 2
        ankle_center_2d = (keypoints_2d[15, :2] + keypoints_2d[16, :2]) / 2

        for i in range(17):
            if i in [15, 16]:
                continue
            z = self.body_height_map[i]
            if keypoints_2d[i, 2] > 0.3:
                dx = keypoints_2d[i, 0] - ankle_center_2d[0]
                dy = keypoints_2d[i, 1] - ankle_center_2d[1]
                pose_3d[i] = [
                    ankle_center_world[0] + dx * self.scale_factor * z,
                    ankle_center_world[1] + dy * self.scale_factor * z,
                    z
                ]
            else:
                pose_3d[i] = [ankle_center_world[0], ankle_center_world[1], z]

        return pose_3d

    def temporal_smoothing(self, pose_3d, player_id, alpha=0.3):
        if player_id in self.previous_poses:
            smoothed = alpha * pose_3d + (1 - alpha) * self.previous_poses[player_id]
        else:
            smoothed = pose_3d
        self.previous_poses[player_id] = pose_3d.copy()
        return smoothed


def process_json_file(json_path, output_path):
    print("=" * 60)
    print("3D POSE ESTIMATION — STANDARD GEOMETRIC METHOD")
    print("=" * 60)
    print(f"Input file : {json_path}")
    print(f"Output file: {output_path}")
    print()
    print("HOW IT WORKS:")
    print("  - Ankles are anchored to their known real-world court position")
    print("  - Each joint is assigned a fixed height (e.g. nose=1.70m, hip=0.95m)")
    print("  - Pixel offset from ankle → scaled to meters using scale factor")
    print("  - Result is smoothed across frames to reduce jitter")
    print("=" * 60)
    print()

    print("Loading data...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"✓ Loaded successfully")
    print(f"  Video     : {data['video_info']['video_name']}")
    print(f"  Frames    : {data['video_info']['frame_count']}")
    print(f"  Resolution: {data['video_info']['width']}x{data['video_info']['height']}")
    print(f"  Court     : {data['court_info']['width_meters']}m x {data['court_info']['length_meters']}m")
    print()

    estimator = PoseEstimator3D()

    results = {
        'video_info': data['video_info'],
        'court_info': data['court_info'],
        'poses_3d': {}
    }

    total_frames = len(data['frame_data'])
    processed_count = 0
    failed_count = 0

    print(f"Processing {total_frames} frames...")
    print("-" * 60)

    start_time = time.time()

    for idx, (frame_id, frame_data) in enumerate(data['frame_data'].items(), 1):
        if idx % 50 == 0 or idx == 1:
            elapsed = time.time() - start_time
            fps = idx / elapsed if elapsed > 0 else 0
            print(f"  Frame {idx}/{total_frames}  (ID: {frame_id})  —  {fps:.1f} fps")

        results['poses_3d'][frame_id] = {}

        for player_id, player_data in frame_data.items():
            if 'keypoints_2d' not in player_data or 'ankles' not in player_data:
                failed_count += 1
                continue

            if len(player_data['ankles']) < 2:
                failed_count += 1
                continue

            keypoints_2d = [
                [kpt['x'], kpt['y'], kpt['confidence']]
                for kpt in player_data['keypoints_2d']
            ]

            if len(keypoints_2d) != 17:
                failed_count += 1
                continue

            ankle_world = [
                [player_data['ankles'][0]['world_x'], player_data['ankles'][0]['world_y']],
                [player_data['ankles'][1]['world_x'], player_data['ankles'][1]['world_y']]
            ]

            try:
                pose_3d = estimator.estimate_pose(np.array(keypoints_2d), ankle_world)
                pose_3d = estimator.temporal_smoothing(pose_3d, player_id)

                results['poses_3d'][frame_id][player_id] = {
                    'joints_3d': pose_3d.tolist(),
                    'joint_names': [
                        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
                    ]
                }
                processed_count += 1

            except Exception as e:
                if idx == 1:
                    print(f"  [ERROR] Frame {frame_id}, {player_id}: {e}")
                failed_count += 1

    elapsed = time.time() - start_time

    print("-" * 60)
    print()
    print("SUMMARY")
    print("-" * 60)
    print(f"  Method                   : Standard Geometric Estimation")
    print(f"✓ Total frames processed   : {total_frames}")
    print(f"✓ Successful estimations   : {processed_count}")
    print(f"✓ Processing time          : {elapsed:.2f}s  ({total_frames / elapsed:.1f} fps)")
    if failed_count > 0:
        print(f"⚠ Failed estimations       : {failed_count}")
    print()

    print(f"Saving results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Saved successfully!")
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    BASE = os.path.join(os.path.dirname(__file__), "..")
    input_file = os.path.join(BASE, "data", "input", "positions.json")
    output_file = os.path.join(BASE, "data", "output", "poses_3d_output.json")

    process_json_file(input_file, output_file)
