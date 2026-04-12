import threading
import time

def stress_cpu():
    while True:
        x = sum(i * i for i in range(100000))

threads = []
for _ in range(4):
    t = threading.Thread(target=stress_cpu, daemon=True)
    t.start()
    threads.append(t)

print("✅ CPU stress running — press Ctrl+C to stop")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopped")