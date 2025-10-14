import os
from datetime import datetime


class Logger:
    def __init__(self, save_path, timestamp, extend=''):
        self.save_path = os.path.join(save_path, timestamp, f'{timestamp}.txt')
        if extend != '':
            self.save_path = os.path.join(save_path, timestamp, f'{timestamp}_{extend}.txt')
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        os.makedirs(os.path.join(save_path, timestamp, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(save_path, timestamp, "imgs"), exist_ok=True)

    def record(self, logs, print_flag=True):
        with open(self.save_path, "a", encoding="utf-8") as log_file:
            for log in logs:
                log = datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ") + log
                log_file.write(log + "\n")
                if print_flag:
                    print(log)