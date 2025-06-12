import  cv2
import  logging
import  numpy as np
from    _utility        import get_center_from_bbox
from    II_tracking     import Feature
from    YOLO_inference  import detect_horse_features, model

log = logging.getLogger(__name__)

def maintenance_inference(
        frame: np.ndarray,
        features: list[Feature],
        model_confidence=0.4,
        frame_i=None,
        big_move_threshold=80,
        max_update_dist=1920
    ) -> None:
    _, new_objects = detect_horse_features(model, frame, model_confidence=model_confidence)
    updated_count = 0
    unused_objects = list(new_objects)

    for feature in features.values():
        feature_class = feature.label.split("_")[1] if "_" in feature.label else feature.label

        candidates = [obj for obj in unused_objects if (obj['label'] if isinstance(obj, dict) else obj.label) == feature_class]
        if not candidates:
            feature.was_reset = False
            continue

        feature_center = get_center_from_bbox(feature.bbox)
        min_dist = float('inf')
        closest_obj = None

        for obj in candidates:
            obj_bbox = obj['bbox'] if isinstance(obj, dict) else obj.bbox
            obj_center = get_center_from_bbox(obj_bbox)
            dist = np.linalg.norm(obj_center - feature_center)
            if dist < min_dist:
                min_dist = dist
                closest_obj = obj

        if closest_obj is not None and min_dist < max_update_dist:
            new_bbox = closest_obj['bbox'] if isinstance(closest_obj, dict) else closest_obj.bbox
            move_distance = np.linalg.norm(get_center_from_bbox(new_bbox) - feature_center)
            feature.bbox = list(new_bbox)
            feature.update_point(get_center_from_bbox(new_bbox).reshape(1, 2))
            updated_count += 1

            unused_objects.remove(closest_obj)

            # Mark feature as reset if a big jump
            feature.was_reset = move_distance > big_move_threshold
        else:
            feature.was_reset = False  # Or remove/reset attribute

    log.info(f"[Maintenance Inference] Ran detection on frame {frame_i}: {updated_count}/{len(features)} features updated.")


def find_best_starting_frame(frames: list, model_confidence=0.5, attempts: int = 50, show: bool = False) -> tuple[int, object]:
    best_frame  = {"frame_index": 0, "confidence_sum": 0, "detected_objects": None}
    features    = {}
    
    log.info("Finding the best starting frame")
    for frame_index in range(attempts):
        img, detected_objects = detect_horse_features(model, frames[frame_index], model_confidence=model_confidence)
        
        ears_and_heads = [obj for obj in detected_objects if obj["label"] in ["ear", "head"]]
        confidence_sum = sum([obj["conficence"] for obj in ears_and_heads])

        if confidence_sum > best_frame["confidence_sum"]:
            best_frame["frame_index"]       = frame_index
            best_frame["confidence_sum"]    = confidence_sum
            #best_frame["detected_objects"]  = ears_and_heads
            best_frame["detected_objects"]  = detected_objects
            best_frame["img"]               = img
    
    log.info(f"Best starting frame found: Frame {best_frame['frame_index']} with a combined ear+head confidence of {round(best_frame['confidence_sum'],4)} - Found {len(best_frame["detected_objects"])} features")
    if show:
        cv2.imshow(f'Detections on best image', best_frame["img"])
        cv2.waitKey(0)

    for i, feature in enumerate(best_frame["detected_objects"]):
        #if feature["label"] in ["head", "ear"]:
            features[i] = Feature(feature["bbox"], f"{i}_{feature['label']}")

    ears  = len([feature for _, feature in features.items() if "ear"  in feature.label])
    heads = len([feature for _, feature in features.items() if "head" in feature.label])

    if ears == 0 or heads == 0:
        raise Exception(f"Head and ears not found in {attempts} attempts. Exiting")

    return best_frame["frame_index"], features