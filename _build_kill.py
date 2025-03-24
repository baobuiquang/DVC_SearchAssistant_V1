import _build_config
import psutil

def kill_process(process_name):
    print(f"> Finding process '{process_name}'")

    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            proc_name = proc.info['name']
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            if (process_name.endswith('.py') and 'python' in proc_name and process_name in cmdline) or \
               (process_name.endswith('.exe') and process_name == proc_name):

                proc.terminate()
                proc.wait()
                print(f"> Terminated process with PID={proc.info['pid']} ({proc_name})")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

kill_process(f"{_build_config.PYTHON_FILE_NAME}")
kill_process(f"{_build_config.BUILD_NAME}.exe")