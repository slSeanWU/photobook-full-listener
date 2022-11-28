import os
import json
from processor import Log


def load_logs(log_repository, data_path):

    filepath = os.path.join(data_path, log_repository)
    print("Loading logs from {}...".format(filepath))

    file_count = 0
    for _, _, files in os.walk(filepath):
        file_count += len(files)
    print("{} files found.".format(file_count))
    logs = []
    for root, dirs, files in os.walk(filepath):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file), 'r') as logfile:
                    log = Log(json.load(logfile))
                    if log.complete:
                        logs.append(log)

    print("DONE. Loaded {} completed game logs.".format(len(logs)))
    return logs
