import os
import sys
from datetime import datetime

class Tee:
    def __init__(self, file_path, mode="w"):
        self.file = open(file_path, mode, buffering=1)
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def __enter__(self):
        sys.stdout = self
        sys.stderr = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        self.file.close()
        sys.stdout = self.stdout
        sys.stderr = self.stdout