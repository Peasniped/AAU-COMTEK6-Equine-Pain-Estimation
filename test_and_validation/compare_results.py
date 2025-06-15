import os
import pandas as pd
import re

EQUIFACS_CSV = os.path.join("test_and_validation", "eqifacs_results_ead.csv")
TIMING_LEEWAY = 25/4  # Maximum frame discrepancy for a correct match
NEAR_MISS_LEEWAY = 25 # Maximum Frame discrepancy for a near miss

def read_events(file):
    return pd.read_csv(file, sep=",")

def normalize_code(code):
    # Strip trailing L/R, e.g. EAD101L -> EAD101
    return re.sub(r'[LR]$', '', code)

def codes_match(code1, code2):
    return normalize_code(str(code1)) == normalize_code(str(code2))

def intervals_match(start1, end1, start2, end2, leeway):
    # Returns True if intervals overlap OR are within leeway
    return not (end1 + leeway < start2 or end2 + leeway < start1)

def compare_events(equifacs_df, my_df):
    results = {}
    videos = set(equifacs_df['video_name']).intersection(set(my_df['video_name']))

    for video in videos:
        equifacs_events = equifacs_df[equifacs_df['video_name'] == video]
        my_events = my_df[my_df['video_name'] == video]

        correct = 0
        near_miss = 0
        correct_timing_deviations = []
        false_negatives = 0
        equifacs_used = set()
        my_used = set()

        # Compare each equifacs event to my events
        for i, equifacs_row in equifacs_events.iterrows():
            matched = False
            close_but_not_quite = False
            for j, my_row in my_events.iterrows():
                if codes_match(my_row['code'], equifacs_row['code']):
                    # Check if within perfect leeway (correct)
                    if intervals_match(
                        equifacs_row['start_frame'], equifacs_row['end_frame'],
                        my_row['start_frame'], my_row['end_frame'],
                        leeway=TIMING_LEEWAY
                    ):
                        matched = True
                        equifacs_used.add(i)
                        my_used.add(j)
                        # Timing deviation: mean of abs start and end differences
                        start_dev = abs(equifacs_row['start_frame'] - my_row['start_frame'])
                        end_dev = abs(equifacs_row['end_frame'] - my_row['end_frame'])
                        correct_timing_deviations.append((start_dev + end_dev) / 2)
                        break
                    # If not correct, check near miss
                    elif intervals_match(
                        equifacs_row['start_frame'], equifacs_row['end_frame'],
                        my_row['start_frame'], my_row['end_frame'],
                        leeway=NEAR_MISS_LEEWAY
                    ):
                        close_but_not_quite = True
                        # Don't break, look for a possible perfect match

            if matched:
                correct += 1
            elif close_but_not_quite:
                near_miss += 1
            else:
                false_negatives += 1

        # False positives: my events not matched to any equifacs event
        false_positives = len(my_events) - len(my_used)
        
        total_equifacs = len(equifacs_events)
        percent = (correct / total_equifacs * 100) if total_equifacs > 0 else 0
        avg_deviation = sum(correct_timing_deviations) / len(correct_timing_deviations) if correct_timing_deviations else 0

        fn_and_nm = false_negatives + near_miss
        precision = round(correct / (correct + false_positives)           if (correct + false_positives) > 0  else 0.0, 3)
        recall    = round(correct / (correct + fn_and_nm)                 if (correct + fn_and_nm) > 0       else 0.0, 3)
        f1score   = round(2 * (precision * recall) / (precision + recall) if (precision + recall) > 0         else 0.0, 3)

        results[video] = {
            "correct":          correct,
            "near_miss":        near_miss,
            "total":            total_equifacs,
            "percent":          percent,
            "false_negatives":  false_negatives,
            "false_positives":  false_positives,
            "avg_deviation":    avg_deviation,
            "precision":        precision,
            "recall":           recall,
            "f1score":          f1score
        }
    return results

def natural_sort_key(s):
    # Splits string into list of strings and integers: 'S12file3' -> ['S', 12, 'file', 3]
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]

def print_summary(results):
    # Initialize aggregate stats
    grand_total = {
        "correct": 0,
        "near_miss": 0,
        "total": 0,
        "false_negatives": 0,
        "false_positives": 0,
        "correct_timing_deviations": [],
    }

    for video in sorted(results.keys(), key=natural_sort_key):  # <-- natural sort
        stats = results[video]
        print(f"Video: {video}")
        print(f"  Correct events: {stats['correct']} of {stats['total']} ({stats['percent']:.1f}%)")
        print(f"  Near misses (almost correct): {stats['near_miss']}")
        print(f"  False negatives: {stats['false_negatives']}")
        print(f"  False positives: {stats['false_positives']}")
        if stats['correct'] > 0:
            print(f"  Avg timing deviation (frames): {stats['avg_deviation']:.2f}")
        print(f"  -------------------------")
        print(f"  Precision: {stats['precision']}")
        print(f"  Recall: {stats['recall']}")
        print(f"  F1 Score: {stats['f1score']}")
        print()

        # Aggregate for totals
        grand_total["correct"] += stats["correct"]
        grand_total["near_miss"] += stats["near_miss"]
        grand_total["total"] += stats["total"]
        grand_total["false_negatives"] += stats["false_negatives"]
        grand_total["false_positives"] += stats["false_positives"]
        if stats['correct'] > 0:
            grand_total["correct_timing_deviations"].extend(
                [stats['avg_deviation']] * stats['correct']
            )

    if grand_total["total"] > 0:
        overall_percent = (grand_total["correct"] / grand_total["total"]) * 100
    else:
        overall_percent = 0.0
    if grand_total["correct_timing_deviations"]:
        overall_avg_deviation = sum(grand_total["correct_timing_deviations"]) / len(grand_total["correct_timing_deviations"])
    else:
        overall_avg_deviation = 0.0

    correct         = grand_total["correct"]
    false_positives = grand_total["false_positives"]
    fn_and_nm       = grand_total["false_negatives"] + grand_total["near_miss"]

    precision = round(correct / (correct + false_positives)           if (correct + false_positives) > 0 else 0.0, 3)
    recall    = round(correct / (correct + fn_and_nm)                 if (correct + fn_and_nm) > 0       else 0.0, 3)
    f1score   = round(2 * (precision * recall) / (precision + recall) if (precision + recall) > 0        else 0.0, 3)

    grand_total["precision"] = precision
    grand_total["recall"]    = recall
    grand_total["f1score"]   = f1score

    print("=== Grand Total Across All Videos ===")
    print(f"  Total correct events: {grand_total['correct']} of {grand_total['total']} ({overall_percent:.1f}%)")
    print(f"  Total near misses: {grand_total['near_miss']}")
    print(f"  Total false negatives: {grand_total['false_negatives']}")
    print(f"  Total false positives: {grand_total['false_positives']}")
    if grand_total['correct'] > 0:
        print(f"Overall avg timing deviation (frames): {overall_avg_deviation:.2f}")
    print(f"  -------------------------")
    print(f"  Precision: {grand_total['precision']}")
    print(f"  Recall: {grand_total['recall']}")
    print(f"  F1 Score: {grand_total['f1score']}")
    print()

def compare_to_equifacs(my_csv_name: str) -> None:
    my_csv_path = os.path.join("test_and_validation", my_csv_name)
    my_csv_df   = read_events(my_csv_path)
    equifacs_df = read_events(EQUIFACS_CSV)
    results     = compare_events(equifacs_df, my_csv_df)
    print_summary(results)

if __name__ == "__main__":
    csv_name = "results_2025-06-15_01.csv"
    compare_to_equifacs(csv_name)
