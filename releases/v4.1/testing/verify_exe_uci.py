import subprocess
import sys
import textwrap
import time

exe = r"dist\V7P3R_v4.1.exe"

cmd = [exe]

proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

# Send UCI startup commands similar to Arena
commands = textwrap.dedent("""
uci
isready
position startpos
go movetime 1000
quit
""")

try:
    out, _ = proc.communicate(commands, timeout=15)
except subprocess.TimeoutExpired:
    proc.kill()
    out, _ = proc.communicate()

print(out)

# Basic assertions (print results instead of raising)
print('HAS id name:', 'id name' in out)
print('HAS uciok:', 'uciok' in out)
print('HAS readyok:', 'readyok' in out)
print('HAS bestmove:', 'bestmove' in out)
