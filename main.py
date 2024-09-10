import subprocess

def run_script(script_name):
    """Run a script using subprocess."""
    result = subprocess.run(['python', script_name], capture_output=True, text=True)
    print(f"Running {script_name}...")
    print("Output:", result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    result.check_returncode()

def main():
    try:
        # Run the scripts in order
        run_script('tweets_extraction.py')
        run_script('pre-processing.py')
        run_script('processing.py')
        run_script('folder_statistics.py')  # Assuming this is a script for statistics
        run_script('mongo_db.py')  # Assuming this is the script interacting with MongoDB

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the scripts: {e}")

if __name__ == "__main__":
    main()
