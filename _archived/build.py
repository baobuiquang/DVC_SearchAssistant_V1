import _config
import PyInstaller.__main__
import shutil
import os

BUILD_DIR = "build"

PyInstaller.__main__.run([
    f'{_config.PYTHON_FILE_NAME}',
    f'--name={_config.BUILD_NAME}',
    '--noconsole',
    '--clean',
    '--noconfirm',
    '--onedir',
    '--log-level=WARN',
    f'--distpath=./{BUILD_DIR}/dist',
    f'--workpath=./{BUILD_DIR}/temp',
    f'--specpath=./{BUILD_DIR}',
    '--contents-directory=bin',
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
for fld in _config.FOLDERS_TO_ADD:
    copy_folder(f"{fld}", f"{BUILD_DIR}/dist/{_config.BUILD_NAME}/{fld}")