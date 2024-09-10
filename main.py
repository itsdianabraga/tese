import subprocess

def run_script(script_name):
    """Run a script using subprocess."""
    result = subprocess.run(['python', script_name], capture_output=True, text=True)
    print(f"Running {script_name}...")
    if result.stderr:
        print("Errors:", result.stderr)
    result.check_returncode()

def main():
    try:
        # Run the scripts in order
        run_script('./tweets_data/tweets_extraction.py')
        run_script('./data_clean/pre-processing.py')
        run_script('./data_clean/processing.py')
        run_script('./statistics/each_category.py')
        run_script('./statistics/global.py')
        run_script('./statistics/links.py')
        run_script('./mongo_conection/mongo_db.py')  # Assuming this is the script interacting with MongoDB

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the scripts: {e}")

if __name__ == "__main__":
    main()
