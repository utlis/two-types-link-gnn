import argparse
import sys
import csv
import os
from datetime import datetime

def load_terms(file_path):
    """
    Reads terms from a text file and returns them as a set.
    Performs whitespace stripping and ignores empty lines.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Strip whitespace, convert to lowercase, and add to set (case-insensitive)
            terms = {line.strip().lower() for line in f if line.strip()}
        return terms
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: An error occurred while reading file '{file_path}'.\n{e}")
        sys.exit(1)

def calculate_metrics(ground_truth_path, result_path):
    # Load files
    gt_terms = load_terms(ground_truth_path)
    result_terms = load_terms(result_path)

    # Calculate Intersection (True Positives)
    intersection = gt_terms.intersection(result_terms)
    true_positives = len(intersection)

    len_gt = len(gt_terms)
    len_result = len(result_terms)

    print("--- Statistics ---")
    print(f"Number of terms in Ground Truth: {len_gt}")
    print(f"Number of terms in Result      : {len_result}")
    print(f"Number of matching terms (TP)  : {true_positives}")
    print("------------------")

    # Precision = TP / (TP + FP) = TP / len_result
    if len_result > 0:
        precision = true_positives / len_result
    else:
        precision = 0.0

    # Recall = TP / (TP + FN) = TP / len_gt
    if len_gt > 0:
        recall = true_positives / len_gt
    else:
        recall = 0.0

    # F1-score = 2 * (Precision * Recall) / (Precision + Recall)
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    return {
        "len_gt": len_gt,
        "len_result": len_result,
        "true_positives": true_positives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

def save_to_csv(output_path, gt_path, res_path, metrics):
    """
    Appends calculation results to a CSV file.
    Creates a header if the file does not exist.
    """
    file_exists = os.path.isfile(output_path)
    
    header = [
        "timestamp", 
        "ground_truth_file", 
        "result_file", 
        "len_gt", 
        "len_result", 
        "true_positives", 
        "precision", 
        "recall", 
        "f1_score"
    ]
    
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        gt_path,
        res_path,
        metrics["len_gt"],
        metrics["len_result"],
        metrics["true_positives"],
        f"{metrics['precision']:.4f}",
        f"{metrics['recall']:.4f}",
        f"{metrics['f1_score']:.4f}"
    ]
    
    try:
        with open(output_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)
        print(f"Results output to CSV file: {output_path}")
    except Exception as e:
        print(f"Error occurred while writing to CSV file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare two text files and calculate Recall, Precision, and F1-score, then output to CSV.')
    
    # Argument settings
    parser.add_argument('--gt', default='ground_truth.txt', help='Path to Ground Truth file')
    parser.add_argument('--res', default='result.txt', help='Path to Result file')
    parser.add_argument('--out', default='metrics.csv', help='Path to output CSV file (default: metrics.csv)')

    args = parser.parse_args()

    print(f"Ground Truth file: {args.gt}")
    print(f"Result file      : {args.res}")
    print("")

    metrics = calculate_metrics(args.gt, args.res)

    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1_score']:.4f}")
    print("")

    save_to_csv(args.out, args.gt, args.res, metrics)
