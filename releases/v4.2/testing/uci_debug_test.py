import subprocess
import sys
import time

# Interactive UCI test to debug why engine isn't making moves
# This sends commands one by one and shows responses with timing

exe_path = r".\v7p3r_uci.py"

proc = subprocess.Popen([sys.executable, exe_path], 
                       stdin=subprocess.PIPE, 
                       stdout=subprocess.PIPE, 
                       stderr=subprocess.PIPE, 
                       text=True, 
                       bufsize=1)

def send_and_wait(cmd, timeout=5):
    print(f">>> {cmd}")
    proc.stdin.write(cmd + "\n")
    proc.stdin.flush()
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        # Try to read a line non-blocking
        if proc.poll() is None:  # Process still running
            line = proc.stdout.readline()
            if line:
                print(f"<<< {line.strip()}")
                if line.strip() in ['uciok', 'readyok'] or line.strip().startswith('bestmove'):
                    return line.strip()
        time.sleep(0.01)
    
    print(f"No response within {timeout}s")
    return None

print("=== UCI Debug Test ===")

# Test basic UCI startup
send_and_wait("uci", 2)
send_and_wait("isready", 2)

# Test position setup and search with various time controls
print("\n=== Testing position and go commands ===")
send_and_wait("position startpos", 1)

# Test with movetime (should be fast)
print("\nTesting: go movetime 500")
start = time.time()
response = send_and_wait("go movetime 500", 3)
elapsed = time.time() - start
print(f"Response time: {elapsed:.2f}s")

# Test with depth
print("\nTesting: go depth 3")
send_and_wait("position startpos", 1)
start = time.time()
response = send_and_wait("go depth 3", 5)
elapsed = time.time() - start
print(f"Response time: {elapsed:.2f}s")

# Test with wtime/btime (typical Arena usage)
print("\nTesting: go wtime 300000 btime 300000")
send_and_wait("position startpos", 1)
start = time.time()
response = send_and_wait("go wtime 300000 btime 300000", 5)
elapsed = time.time() - start
print(f"Response time: {elapsed:.2f}s")

send_and_wait("quit", 1)
proc.terminate()
