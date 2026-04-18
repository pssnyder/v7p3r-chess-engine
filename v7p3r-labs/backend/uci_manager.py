"""
UCI Engine Manager - Handles Docker container lifecycle and UCI communication
Adapted from lichess-bot pattern with Docker containerization
"""
import asyncio
import subprocess
from typing import Optional
import logging
from models import EngineVersion

logger = logging.getLogger(__name__)


class UCIEngine:
    """Manages a single UCI engine instance in Docker container"""
    
    def __init__(self, version: EngineVersion, game_id: str):
        self.version = version
        self.game_id = game_id
        self.container_name = f"v7p3r-{version.value.replace('.', '_')}-{game_id}"
        self.process: Optional[subprocess.Popen] = None
        self.running = False
        
    async def start(self):
        """Start Docker container with engine"""
        try:
            # Docker run command (ephemeral container with resource limits)
            cmd = [
                "docker", "run", "--rm", "-i",
                f"--name={self.container_name}",
                "--memory=256m",
                "--cpus=0.3",
                f"gcr.io/rts-labs-f3981/v7p3r:{self.version.value}"
            ]
            
            logger.info(f"Starting engine {self.version.value} for game {self.game_id}")
            
            # Spawn process with stdin/stdout pipes (lichess-bot pattern)
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Initialize UCI protocol
            await self._send_command("uci")
            
            # Wait for uciok
            while True:
                line = await self._read_line()
                if line == "uciok":
                    break
                    
            self.running = True
            logger.info(f"Engine {self.version.value} started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start engine {self.version.value}: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop engine and clean up container"""
        if self.process:
            try:
                # Send quit command
                if self.running:
                    await self._send_command("quit")
                    await asyncio.sleep(0.1)
                
                # Terminate process
                self.process.terminate()
                
                # Wait for container to stop (Docker --rm will auto-remove)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    
                logger.info(f"Engine {self.version.value} stopped")
                
            except Exception as e:
                logger.error(f"Error stopping engine {self.version.value}: {e}")
            finally:
                self.process = None
                self.running = False
    
    async def get_move(self, fen: str, wtime_ms: int, btime_ms: int, 
                       winc_ms: int, binc_ms: int) -> Optional[str]:
        """
        Get best move from engine
        Args:
            fen: Position in FEN notation
            wtime_ms: White time remaining in milliseconds
            btime_ms: Black time remaining in milliseconds
            winc_ms: White increment in milliseconds
            binc_ms: Black increment in milliseconds
        Returns:
            Best move in UCI format (e.g., "e2e4")
        """
        if not self.running:
            raise RuntimeError("Engine not running")
        
        try:
            # Set position
            await self._send_command(f"position fen {fen}")
            
            # Request move with time control
            go_cmd = f"go wtime {wtime_ms} btime {btime_ms} winc {winc_ms} binc {binc_ms}"
            await self._send_command(go_cmd)
            
            # Parse response for bestmove
            while True:
                line = await self._read_line()
                if line.startswith("bestmove"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return parts[1]
                    break
                    
            return None
            
        except Exception as e:
            logger.error(f"Error getting move from engine: {e}")
            return None
    
    async def _send_command(self, cmd: str):
        """Send UCI command to engine"""
        if not self.process or not self.process.stdin:
            raise RuntimeError("Engine process not available")
        
        self.process.stdin.write(cmd + "\n")
        self.process.stdin.flush()
        logger.debug(f"Sent to engine: {cmd}")
    
    async def _read_line(self) -> str:
        """Read line from engine stdout"""
        if not self.process or not self.process.stdout:
            raise RuntimeError("Engine process not available")
        
        # Run blocking readline in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        line = await loop.run_in_executor(None, self.process.stdout.readline)
        line = line.strip()
        logger.debug(f"Received from engine: {line}")
        return line
