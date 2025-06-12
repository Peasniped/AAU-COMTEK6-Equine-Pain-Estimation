import  cv2
import  logging
import  numpy as np
import  time

from    I_localization      import find_best_starting_frame, maintenance_inference
from    II_tracking_CNN     import Tracker
import  III_motion_analysis
import  IV_motion_mapping
from    _utility            import setup_root_logger, extract_frames, visualize_frame, get_head_centers_from_motions, export_labelled_segments, _get_csv_filename

setup_root_logger()
log = logging.getLogger(__name__)

def start(video_path: str, show: bool = True, export_csv: bool = False, csv_path: str = None):
    video_name = video_path.split("/")[1].split(".")[0]
    frames = extract_frames(video_path)
    labeled_segments = []
    
    # I
    starting_frame_i = 0
    
    # II
    tracker = Tracker()
    segmenter = III_motion_analysis.MotionSegmenter()

    for frame_i, frame in enumerate(frames[starting_frame_i:], starting_frame_i):
        motions = tracker.track_frame(frame, frame_i)
        head_from, head_to = get_head_centers_from_motions(motions)
        segmenter.last_head_to = head_to

    # III
        for motion in motions:
            motion['frame'] = frame_i
            feature = motion['feature']
            if "ear" in feature.label:
                if head_to is not None:
                    rel_vector = np.array(motion['to']) - head_to
                    if not hasattr(feature, "rel_trail"):
                        feature.rel_trail = []
                    feature.rel_trail.append(tuple(rel_vector))
                    if len(feature.rel_trail) > 20:
                        feature.rel_trail.pop(0)
            segmenter.update(motion, head_from=head_from, head_to=head_to)

    # IV
            for closed_segment in segmenter.get_just_closed():
                labeled_segment = IV_motion_mapping.label_ear_movement(closed_segment, _print=True)
                labeled_segments.append(labeled_segment)

        if show:
            visualize_frame(
                frames, frame_i, tracker.features, segmenter=segmenter,
                show_trails=True, show_bbox=True, show_labels=True,
                show_active_segments=True, show_completed_segments=False,
                show_relative_trails=True, video_name=video_name
            )
            
            key = cv2.waitKey(30)
            if key == ord("q"):
                log.warning("Pressed q to quit playback! Exiting...\n")
                exit()
            if key == ord("s"):
                log.warning("Pressed s to skip to next video!\n")
                break
    
    log.info(f"Done tracking movments in {video_path}")
    export_labelled_segments(labeled_segments, video_name, to_csv=export_csv, csv_path=csv_path)
    return len(frames)

def do_test(video_numbers: list[int] | int | None = None, export_csv: bool = False) -> None:
    fps_sum = 0
    fps_strings = []
    if video_numbers and type(video_numbers) == int:
        video_numbers = [video_numbers]
    elif type(video_numbers) == list:
        video_numbers = video_numbers
    else:
        video_numbers = list(range(1,13))
    csv_path = _get_csv_filename() if export_csv else None

    log.info(f"---- Starting new run on videos: {video_numbers}")
    for video_num in video_numbers:
        file = f"video/S{video_num}_Video.mp4"
        log.info(f"Testing {file}")
        start_time = time.perf_counter()
        frames = start(file, export_csv=export_csv, csv_path=csv_path)
        seconds = time.perf_counter() - start_time
        fps = frames / seconds
        fps_string = f"{file}: with {frames} frames in {seconds} seconds -- fps: {fps}"
        log.info(fps_string)
        fps_strings.append(fps_string)
        fps_sum += fps
    
    videos = len(video_numbers)
    avg_fps = fps_sum / videos
    log.info(f"Concluded testing of {videos} videos, with an average fps of {avg_fps}")

if __name__ == "__main__":
    do_test(export_csv = True)