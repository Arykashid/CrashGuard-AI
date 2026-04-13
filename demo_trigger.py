import threading
import time
import multiprocessing

def stress_cpu():
    while True:
        x = [i**2 + i**3 for i in range(1000000)]

def main():
    print("=" * 50)
    print("  CrashGuard AI — Strong Demo Trigger")
    print("=" * 50)
    
    # Use ALL cores
    num_threads = multiprocessing.cpu_count()
    print(f"  Starting stress on {num_threads} threads...")
    
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=stress_cpu, daemon=True)
        t.start()
        threads.append(t)
        print(f"  ✅ Thread {i+1}/{num_threads} started")
    
    print("\n  🔥 Max CPU stress running!")
    print("  Watch Current CPU on dashboard rise above 70%")
    print("  Press Ctrl+C to stop\n")
    
    try:
        elapsed = 0
        while True:
            time.sleep(5)
            elapsed += 5
            print(f"  ⏱️ Running {elapsed}s — check dashboard CPU %")
    except KeyboardInterrupt:
        print("\n  ✅ Stopped")

if __name__ == "__main__":
    main()