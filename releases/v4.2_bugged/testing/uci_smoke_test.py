import subprocess
import sys
import textwrap

# This smoke test launches the engine in unbuffered mode (-u), sends a batch of UCI
# commands including a final 'quit', and uses communicate() to collect output so
# the test can't block waiting on interactive input.

engine_cmd = [sys.executable, "-u", "-m", "v7p3r_uci"]

proc = subprocess.Popen(engine_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

commands = textwrap.dedent(
    """
    uci
    isready
    position startpos
    go movetime 100
    quit
    """
)

try:
    out, _ = proc.communicate(commands, timeout=10)
except subprocess.TimeoutExpired:
    proc.kill()
    out, _ = proc.communicate()

print(out)

if "id name" not in out:
    print("Failed: no 'id name' in output")
    sys.exit(1)
if "uciok" not in out:
    print("Failed: no 'uciok' in output")
    sys.exit(1)
if "readyok" not in out:
    print("Failed: no 'readyok' in output")
    sys.exit(1)
if "bestmove" not in out:
    print("Failed: no 'bestmove' in output")
    sys.exit(1)

print("Smoke test passed")
