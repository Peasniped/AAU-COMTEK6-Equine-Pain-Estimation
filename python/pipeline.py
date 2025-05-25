import  cv2
import numpy as np

from    I_localization      import find_best_starting_frame, maintenance_inference
from    II_tracking         import Tracker
import  III_motion_analysis
import  IV_motion_mapping
from    _utility            import extract_frames, visualize_frame, get_head_centers_from_motions

def start(video_path: str, show=True):
    frames = extract_frames(video_path)
    
    # I
    starting_frame_i, features = find_best_starting_frame(frames, attempts=50)
    
    # II
    tracker = Tracker(features)
    segmenter = III_motion_analysis.MotionSegmenter()

    for frame_i, frame in enumerate(frames[starting_frame_i:], starting_frame_i):
        if frame_i % 50 == 0:
            maintenance_inference(frame, tracker.features, frame_i=frame_i)

        motions = tracker.track_frame(frame)
        head_from, head_to = get_head_centers_from_motions(motions)
        segmenter.last_head_to = head_to

    # III
        for motion in motions:
            motion['frame'] = frame_i

            # Compute and store relative to head (for trail visualization)
            feature = motion['feature']
            if "ear" in feature.label:
                if head_to is not None:
                    rel_vec = np.array(motion['to']) - head_to
                    if not hasattr(feature, "rel_trail"):
                        feature.rel_trail = []
                    feature.rel_trail.append(tuple(rel_vec))
                    # Optionally keep only the last N points:
                    if len(feature.rel_trail) > 20:
                        feature.rel_trail.pop(0)
            segmenter.update(motion, head_from=head_from, head_to=head_to)

    # IV
            for closed_segment in segmenter.get_just_closed():
                labeled_segment = IV_motion_mapping.label_ear_movement(closed_segment, _print=True)

        if show:
            visualize_frame(
                frame, frame_i, tracker.features, segmenter=segmenter,
                show_trails=True, show_bbox=True, show_labels=True,
                show_active_segments=True, show_completed_segments=False,
                show_relative_trails=True
            )
            
            key = cv2.waitKey(30)
            if key == ord("q"):
                print("\nPressed q to quit playback! Exiting...\n")
                break
    
    print("Done.")

if __name__ == "__main__":
    start("video/S12_Video.mp4")