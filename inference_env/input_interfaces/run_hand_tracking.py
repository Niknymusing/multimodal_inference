import subprocess
import os

def run_shell_script():
    # Define the directory where the shell script is located
    script_directory = "/Users/nikny/mediapipe"
    
    # Change the current working directory to the script directory
    os.chdir(script_directory)
    
    # Define the path to the shell script
    script_path = "./execute_hand_tracking.sh"
    
    # Run the shell script
    try:
        process = subprocess.Popen(
            script_path, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        
        # Output the subprocess's output
        stdout, stderr = process.communicate()
        print("Script output:", stdout)
        if stderr:
            print("Script errors:", stderr)
    except subprocess.CalledProcessError as e:
        print("An error occurred while running the script:", e)

if __name__ == "__main__":
    run_shell_script()
