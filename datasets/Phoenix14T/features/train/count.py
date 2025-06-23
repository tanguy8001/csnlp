#!/usr/bin/env python3
import os

def count_elements(folder_path, count_files=True, count_dirs=True):
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a valid folder.")
        return

    total = 0
    for item in os.listdir(folder_path):
        full_path = os.path.join(folder_path, item)
        if os.path.isfile(full_path) and count_files:
            total += 1
        elif os.path.isdir(full_path) and count_dirs:
            total += 1

    print(f"\nPath: {folder_path}")
    print(f"Total items: {total} ({'files' if count_files else ''}{' and ' if count_files and count_dirs else ''}{'folders' if count_dirs else ''})")

if __name__ == "__main__":
    # 👇 Set your folder path here
    folder_path = '/work/courses/csnlp/Team3/slt/datasets/Phoenix14T/features/train'

    # Choose what to count
    count_files = True
    count_dirs = True

    count_elements(folder_path, count_files, count_dirs)
