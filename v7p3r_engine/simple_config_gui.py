import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox

# Add parent directory to sys.path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

class SimpleConfigGUI:
    """Simplified V7P3R Configuration GUI with just buttons"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("V7P3R Chess Engine Configuration")
        self.root.geometry("500x400")
        
        # Main container frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add a label
        ttk.Label(self.main_frame, text="V7P3R Chess Engine Configuration", font=("Segoe UI", 16)).pack(pady=20)
        
        # Bottom buttons frame for global actions - positioned at the bottom of the window
        self.bottom_frame = ttk.Frame(self.main_frame)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        # Buttons at the bottom with clear styling
        self.run_button = ttk.Button(
            self.bottom_frame, 
            text="Run Chess Game", 
            command=self._run_chess_game,
            padding=(20, 10)
        )
        self.run_button.pack(side=tk.RIGHT, padx=10)
        
        self.exit_button = ttk.Button(
            self.bottom_frame, 
            text="Exit", 
            command=self.root.destroy,
            padding=(20, 10)
        )
        self.exit_button.pack(side=tk.RIGHT, padx=10)
    
    def _run_chess_game(self):
        """Run the chess game with a simple demo configuration"""
        messagebox.showinfo("Run Chess Game", "This would start a chess game with the current configuration.")

if __name__ == '__main__':
    root = tk.Tk()
    app = SimpleConfigGUI(root)
    root.mainloop()
