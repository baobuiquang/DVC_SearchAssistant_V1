# spec: a = Analysis: add this:
#                        module_collection_mode={'gradio': 'py'}

import _build_config
import PyInstaller.__main__
import shutil
import os

BUILD_DIR = "build"

# PyInstaller.__main__.run([
#     f'{_build_config.PYTHON_FILE_NAME}',
#     f'--name={_build_config.BUILD_NAME}',
#     '--noconsole',
#     '--clean',
#     '--noconfirm',
#     '--onedir',
#     '--log-level=WARN',
#     f'--distpath=./{BUILD_DIR}/dist',
#     f'--workpath=./{BUILD_DIR}/temp',
#     f'--specpath=./{BUILD_DIR}',
#     '--contents-directory=bin',
# ])

PyInstaller.__main__.run([
    f'./{BUILD_DIR}/{_build_config.BUILD_NAME}.spec',
    '--clean',
    '--noconfirm',
    '--log-level=WARN',
    f'--distpath=./{BUILD_DIR}/dist',
    f'--workpath=./{BUILD_DIR}/temp',
])


# Copy folders
def copy_folder(source, destination):
    try:
        if not os.path.exists(source):
            print(f"Source folder '{source}' does not exist.")
            return
        shutil.copytree(source, destination)
        print(f"Folder copied successfully from '{source}' to '{destination}'.")
    except FileExistsError:
        print(f"Destination folder '{destination}' already exists.")
    except PermissionError:
        print("Permission denied.")
    except Exception as e:
        print(f"An error occurred: {e}")
for fld in _build_config.FOLDERS_TO_ADD:
    copy_folder(f"{fld}", f"{BUILD_DIR}/dist/{_build_config.BUILD_NAME}/{fld}")
    




# ==========================================================================================

import os
import shutil

def copy_if_not_exists(src, dst):
    """Recursively copy files and folders from src to dst if they do not already exist."""
    if not os.path.exists(dst):
        os.makedirs(dst)
    
    for item in os.listdir(src):
        src_path = os.path.join(src, item)
        dst_path = os.path.join(dst, item)
        
        if os.path.isdir(src_path):
            copy_if_not_exists(src_path, dst_path)
        elif os.path.isfile(src_path):
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
                print(f"Copied: {src_path} -> {dst_path}")
            else:
                print(f"Skipped (already exists): {dst_path}")

copy_if_not_exists("build/build_bin/bin", "build/dist/VNPT_DVC_SearchAPI/bin")