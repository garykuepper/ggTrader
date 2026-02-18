"""Script to export ggTrader TimescaleDB from Docker with progress."""

import subprocess
import sys
import os
from datetime import datetime


def export_db(
    container_name="ggtrader_db", user="ggtrader", db="ggtrader", output_file="ggtrader_dump.bak"
):
    """Runs pg_dump inside docker and streams progress to console."""

    # We use pg_dump -v (verbose) for progress
    # -Fc is custom format (compressed, binary)
    cmd = ["docker", "exec", "-t", container_name, "pg_dump", "-U", user, "-v", "-Fc", db]

    print(f"Starting export of '{db}' from container '{container_name}'...")
    print(f"Target file: {os.path.abspath(output_file)}")
    print("-" * 50)

    try:
        # Start the process
        # stdout goes to the file (binary)
        # stderr goes to the console (progress text)
        with open(output_file, "wb") as f:
            process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.PIPE, text=True, bufsize=1)

            # Stream stderr to console for progress
            while True:
                line = process.stderr.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()

        if process.returncode == 0:
            file_size = os.path.getsize(output_file) / (1024 * 1024)
            print("-" * 50)
            print(f"SUCCESS: Export completed in {output_file}")
            print(f"File Size: {file_size:.2f} MB")
        else:
            print(f"\nERROR: Export failed with exit code {process.returncode}")
            if os.path.exists(output_file) and os.path.getsize(output_file) == 0:
                os.remove(output_file)

    except KeyboardInterrupt:
        print("\nExport interrupted by user.")
        if process:
            process.terminate()
        sys.exit(1)
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Ensure director exists
    os.makedirs("scripts/db", exist_ok=True)

    # Run the export
    export_db()
