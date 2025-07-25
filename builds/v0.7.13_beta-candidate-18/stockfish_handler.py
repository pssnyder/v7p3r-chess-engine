# stockfish_handler.py

"""Stockfish engine interface for V7P3R chess engine."""

import asyncio
import chess
import subprocess
import os
from typing import Optional
from v7p3r_paths import paths

class StockfishHandler:
    """Handler for Stockfish chess engine communication."""
    
    def __init__(self, stockfish_config: dict):
        """Initialize Stockfish handler.
        
        Args:
            stockfish_config: Stockfish configuration dictionary
        """
        self.stockfish_path = stockfish_config.get('stockfish_path', 'stockfish/stockfish-windows-x86-64-avx2.exe')
        self.elo_rating = stockfish_config.get('elo_rating', 1500)
        self.skill_level = stockfish_config.get('skill_level', 20)
        self.depth = stockfish_config.get('depth', 20)
        self.movetime = stockfish_config.get('movetime', 1000)
        
        self.process = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize Stockfish engine process."""
        if self.initialized:
            return
            
        try:
            # Set up process
            stockfish_exe = str(paths.get_resource_path(str(self.stockfish_path)))
            if not os.path.exists(stockfish_exe):
                raise FileNotFoundError(f"Stockfish not found at: {stockfish_exe}")
                
            # Start process
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            self.process = await asyncio.create_subprocess_exec(
                stockfish_exe,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creationflags
            )
            
            # Initialize UCI mode
            await self._send_command("uci")
            await self._send_command("setoption name UCI_LimitStrength value true")
            await self._send_command(f"setoption name UCI_Elo value {self.elo_rating}")
            await self._send_command(f"setoption name Skill Level value {self.skill_level}")
            await self._send_command("isready")
            
            self.initialized = True
            
        except Exception as e:
            print(f"Stockfish initialization error: {e}")
            if self.process:
                self.process.terminate()
            self.process = None
            raise
            
    async def _send_command(self, cmd: str):
        """Send a command to Stockfish.
        
        Args:
            cmd: Command to send
        """
        if not self.process or not self.process.stdin:
            return
            
        self.process.stdin.write(f"{cmd}\n".encode())
        await self.process.stdin.drain()
        
    async def _get_output(self) -> str:
        """Get output from Stockfish.
        
        Returns:
            str: Output from Stockfish
        """
        if not self.process or not self.process.stdout:
            return ""
            
        return (await self.process.stdout.readline()).decode().strip()
        
    async def get_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Get a move from Stockfish.
        
        Args:
            board: Current board position
            
        Returns:
            Optional[chess.Move]: Best move found or None
        """
        if not self.initialized or not self.process:
            return None
            
        # Set position
        await self._send_command(f"position fen {board.fen()}")
        
        # Start search
        await self._send_command(f"go movetime {self.movetime}")
        
        # Get best move
        bestmove = None
        while True:
            output = await self._get_output()
            if output.startswith("bestmove"):
                move_str = output.split()[1]
                try:
                    bestmove = chess.Move.from_uci(move_str)
                    if board.is_legal(bestmove):
                        return bestmove
                except ValueError:
                    pass
                break
                
        return None
        
    async def quit(self):
        """Clean up Stockfish process."""
        if self.process:
            await self._send_command("quit")
            try:
                await asyncio.wait_for(self.process.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                self.process.terminate()
            self.process = None
            self.initialized = False
