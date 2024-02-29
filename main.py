import subprocess

# Define the list of keywords
keywords = ["02/29"]

# Define the path to the log file
log_file_path = "/home/vidkrix/Desktop/work/learn-python/logfile.txt"

# Construct the command string
command = ["./your_script.sh"] + [log_file_path] + keywords

# Execute the command
subprocess.run(command)
