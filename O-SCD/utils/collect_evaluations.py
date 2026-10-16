import os
import glob
import argparse

def main():
    parser = argparse.ArgumentParser(description="Collect evaluation results.")
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.path.dirname(__file__), '..', 'output'),
                        help="Directory containing evaluation results.")
    parser.add_argument("--summary_file", type=str, default="evaluation_summary.txt",
                        help="Name of the summary file to create.")
    args = parser.parse_args()

    output_dir = args.output_dir
    summary_file = os.path.join(output_dir, args.summary_file)

    eval_files = glob.glob(os.path.join(output_dir, '*', '*', 'evaluation_refined.txt'))

    results = []
    total_iou = 0.0
    total_f1 = 0.0
    count = 0

    for eval_path in eval_files:
        print(f'Processing {eval_path}...')
        try:
            with open(eval_path, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    iou = float(lines[0].split(':')[1].strip())
                    f1 = float(lines[1].split(':')[1].strip())
                    scene = os.path.relpath(eval_path, output_dir)
                    # Remove the filename from the path to get the scene name
                    scene = os.path.dirname(scene)
                    results.append((scene, iou, f1))
                    total_iou += iou
                    total_f1 += f1
                    count += 1
                else:
                    print(f"Warning: {eval_path} does not contain enough lines.")
        except Exception as e:
            print(f"Error reading {eval_path}: {e}")

    avg_iou = total_iou / count if count else 0
    avg_f1 = total_f1 / count if count else 0

    with open(summary_file, 'w') as f:
        f.write('Scene\tMean IoU\tMean F1\n')
        for scene, iou, f1 in sorted(results):
            f.write(f'{scene}\t{iou:.6f}\t{f1:.6f}\n')
        f.write('\n')
        f.write(f'Average\t{avg_iou:.6f}\t{avg_f1:.6f}\n')

    print(f'Summary written to {summary_file}')

if __name__ == "__main__":
    main()
