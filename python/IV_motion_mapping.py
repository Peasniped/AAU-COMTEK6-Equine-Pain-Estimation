import logging
import numpy as np

log = logging.getLogger(__name__)

def label_ear_movement(segment, threshold = 8, context_features=None, _print=False) -> dict|None:
    """
    Map ear movement as "EAD101"(left) or "EAD104"(right) (image coordinates).
    """
    start_pt = np.array(segment["trajectory"][0])
    end_pt = np.array(segment["trajectory"][-1])
    delta_x = end_pt[0] - start_pt[0]
    
    if abs(delta_x) < threshold:
        log.debug(f"Movment discarded due to delta_x={abs(delta_x)} < threshold={threshold}")
        return

    if delta_x < 0:     # Leftward movement
        equifacs_code = "EAD101"
    elif delta_x > 0:   # Rightward movement
        equifacs_code = "EAD104"
    else:
        equifacs_code = "none"

    segment["equifacs_code"] = equifacs_code

    if _print:
        direction = "right" if equifacs_code == "EAD104" else "left"
        log.info(f"Movment detected: {segment["feature"]} moved about {int(abs(delta_x))} pixels to the {direction} and was coded as {equifacs_code}")

    return segment