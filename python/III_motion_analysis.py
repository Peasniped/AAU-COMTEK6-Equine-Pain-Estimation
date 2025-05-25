import numpy as np

class MotionSegmenter:
    def __init__(self, grace_period=5, moving_thresh=3.0):
        self.active_segments = {}
        self.completed_segments = []
        self.below_thresh_counter = {}
        self.just_closed_segments = []
        self.grace_period = grace_period
        self.moving_thresh = moving_thresh

    def update(self, motion_dict, head_from=None, head_to=None, _print=False):
        feature     = motion_dict['feature']
        frame_i     = motion_dict['frame']
        from_point  = np.array(motion_dict['from'])
        to_point    = np.array(motion_dict['to'])
        label       = getattr(feature, 'label', '')

        if "ear" not in label:
            return

        # Compute movement relative to head (if possible)
        if head_from is not None and head_to is not None:
            ear_rel_from = from_point - head_from
            ear_rel_to   = to_point   - head_to
            rel_move     = ear_rel_to - ear_rel_from
        else:
            rel_move = to_point - from_point

        dist = np.linalg.norm(rel_move)
        state = getattr(feature, 'state', 'idle')

        if state == 'idle' and dist > self.moving_thresh:
            self.active_segments[feature] = {
                'start': frame_i,
                'trajectory': [tuple(to_point)],
            }
            self.below_thresh_counter[feature] = 0
            feature.state = 'moving'
        elif state == 'moving':
            if dist > self.moving_thresh:
                self.active_segments[feature]['trajectory'].append(tuple(to_point))
                self.below_thresh_counter[feature] = 0
            else:
                self.below_thresh_counter[feature] += 1
                if self.below_thresh_counter[feature] >= self.grace_period:
                    segment = self.active_segments.pop(feature)
                    segment['end'] = frame_i
                    segment["feature"] = feature.label
                    self.completed_segments.append(segment)
                    self.just_closed_segments.append(segment)   # <--- queue for live mapping
                    feature.state = 'idle'
                    del self.below_thresh_counter[feature]
                    t0 = tuple(int(v) for v in segment['trajectory'][0])
                    t1 = tuple(int(v) for v in segment['trajectory'][-1])
                    if _print:
                        print(f"[{feature.label}] Movement frames {segment['start']}-{segment['end']}: {t0} → {t1}")

    def get_just_closed(self):
        """
        Returns and clears the list of segments just closed in this frame.
        """
        segments = self.just_closed_segments[:]
        self.just_closed_segments.clear()
        return segments

    def finish(self):
        # End any still-active segments at the last frame
        for feature, segment in self.active_segments.items():
            segment['end'] = segment['start'] + len(segment['trajectory']) - 1
            self.completed_segments.append(segment)
        self.active_segments.clear()
        self.below_thresh_counter.clear()