import numpy as np

def label_ear_movement(segment, threshold = 8, context_features=None, _print=False):
    """
    Label ear movement as 'left' or 'right' (image coordinates).
    """
    start_pt = np.array(segment['trajectory'][0])
    end_pt = np.array(segment['trajectory'][-1])
    delta_x = end_pt[0] - start_pt[0]
    
    if abs(delta_x) < threshold:
        return

    if delta_x < 0:
        dir_label = 'left'
    elif delta_x > 0:
        dir_label = 'right'
    else:
        dir_label = 'none'

    segment['label'] = dir_label

    if _print:
        print(f"Movment detected: {segment["feature"]} moved about {int(abs(delta_x))} pixels to the {dir_label}")

    return segment

