#!/usr/bin/env python3
import os
import shutil

def move_contents_and_remove_1(main_directory):
    for root, dirs, files in os.walk(main_directory):
        if "1" in dirs:
            folder_1_path = os.path.join(root, "1")
            parent_path = root

            print(f"\nProcessing folder: {folder_1_path}")

            # Move all files from '1' to its parent directory
            for filename in os.listdir(folder_1_path):
                src_path = os.path.join(folder_1_path, filename)
                dst_path = os.path.join(parent_path, filename)

                # Rename file if there's a conflict
                if os.path.exists(dst_path):
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(dst_path):
                        dst_path = os.path.join(parent_path, f"{base}_{counter}{ext}")
                        counter += 1

                shutil.move(src_path, dst_path)
                print(f"Moved: {src_path} → {dst_path}")

            # Remove the now-empty '1' folder
            os.rmdir(folder_1_path)
            print(f"Deleted folder: {folder_1_path}")

if __name__ == "__main__":
    # 👇 Set your main folder path here
    main_folder_path = '/work/courses/csnlp/Team3/slt/datasets/Phoenix14T/features/train'

    if not os.path.isdir(main_folder_path):
        print(f"Error: {main_folder_path} is not a valid directory.")
    else:
        move_contents_and_remove_1(main_folder_path)
        print("\n✅ Done.")

