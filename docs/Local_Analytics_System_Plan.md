# V7P3R Local Analytics System
## Docker Desktop + LLM Analysis + Scheduled Runs

**Date**: December 4, 2025  
**Status**: Design & Implementation  
**Goal**: Run analytics locally with Docker Desktop, LLM-powered insights, automated scheduling

---

## System Overview

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Windows Task Scheduler (or cron)                           │
│  Every Sunday Midnight → docker-compose up                  │
└────────────────────┬────────────────────────────────────────┘
                     ▼
        ┌────────────────────────┐
        │  Docker Compose Start  │
        └────────┬───────────────┘
                 ▼
    ┌────────────────────────────────┐
    │  1. Fetch Games (Lichess API)  │
    │     - Last 7 days              │
    │     - Save to mounted volume   │
    └────────┬───────────────────────┘
             ▼
    ┌────────────────────────────────┐
    │  2. Parallel Analysis          │
    │     - 12 Stockfish workers     │
    │     - Generate JSON reports    │
    └────────┬───────────────────────┘
             ▼
    ┌────────────────────────────────┐
    │  3. LLM Analysis (OpenAI/Local)│  ← NEW
    │     - Review blunder patterns  │
    │     - Identify weak themes     │
    │     - Suggest code improvements│
    └────────┬───────────────────────┘
             ▼
    ┌────────────────────────────────┐
    │  4. Generate Reports           │
    │     - Technical metrics        │
    │     - LLM insights (MD)        │
    │     - Historical comparisons   │
    └────────┬───────────────────────┘
             ▼
    ┌────────────────────────────────┐
    │  5. Save to Local Storage      │
    │     - analytics_reports/       │
    │     - Persistent volume mount  │
    └────────────────────────────────┘
```

### Key Benefits
- **No Cloud Costs**: Everything runs locally
- **LLM Insights**: AI-powered analysis of weaknesses
- **Persistent Storage**: Reports saved to local filesystem
- **Easy Migration**: Later lift-and-shift to cloud
- **Full Control**: Debug, test, iterate quickly

---

## Components

### 1. Docker Compose Setup

**File**: `docker-compose.yml`

```yaml
version: '3.8'

services:
  analytics:
    build: .
    volumes:
      - ./analytics_reports:/workspace/reports
      - ./analytics_data:/workspace/data
      - ./.env:/app/.env:ro
    environment:
      - STOCKFISH_PATH=/usr/local/bin/stockfish
      - DAYS_BACK=7
      - WORKERS=12
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - LLM_PROVIDER=${LLM_PROVIDER:-openai}
    restart: "no"
```

### 2. LLM Integration

**Options**:

#### Option A: OpenAI GPT-4 (Recommended)
- **Cost**: ~$0.01-0.05 per report
- **Quality**: Best insights
- **Setup**: Just need API key

#### Option B: Local LLM (Free)
- **Options**: Ollama (llama3, mistral)
- **Cost**: $0 (uses local CPU/GPU)
- **Quality**: Good for basic analysis
- **Setup**: Install Ollama separately

### 3. Report Structure

**Directory Layout**:
```
analytics_reports/
├── 2025/
│   ├── week_49_2025-12-01/
│   │   ├── technical_report.json      # Raw metrics
│   │   ├── technical_report.md        # Human-readable stats
│   │   ├── llm_analysis.md            # AI insights ← NEW
│   │   ├── action_items.md            # Code review suggestions ← NEW
│   │   └── pgn/
│   │       └── games.pgn
│   └── week_50_2025-12-08/
│       └── ...
└── historical_summary.json
```

---

## LLM Analysis Prompt

### Input to LLM

```json
{
  "week": "2025-12-01 to 2025-12-08",
  "version": "v17.5",
  "games_analyzed": 223,
  "metrics": {
    "win_rate": 51.6,
    "top1_alignment": 47.3,
    "average_cpl": 1342,
    "critical_blunders_per_game": 4.2,
    "blunder_distribution": {
      "opening": 1.2,
      "middlegame": 2.1,
      "endgame": 0.9
    }
  },
  "top_blunder_positions": [
    {
      "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
      "move_played": "Bc4",
      "best_move": "d4",
      "eval_loss": -180,
      "frequency": 12
    }
  ],
  "historical_comparison": {
    "vs_v17_1": {
      "top1_change": +3.3,
      "cpl_change": -156,
      "blunder_change": -40
    }
  }
}
```

### LLM Prompt Template

```
You are a chess engine development advisor analyzing the performance of V7P3R chess engine.

Review the following weekly performance data and provide:

1. **Strengths Identified**: What is the engine doing well this week?
2. **Weakness Themes**: What patterns of mistakes appear repeatedly?
3. **Opening Issues**: Any specific opening lines causing problems?
4. **Middlegame Tactical Gaps**: Missed tactics or position evaluation errors?
5. **Endgame Problems**: Specific endgame types that need improvement?
6. **Code Review Suggestions**: Concrete recommendations for the next version:
   - Search improvements (depth, pruning, extensions)
   - Evaluation function tweaks (PST, material, mobility)
   - Time management adjustments
   - Bug fixes for specific positions

Format your response as a clear, actionable Markdown document.

Data:
{json_data}
```

### Example LLM Output

```markdown
# V7P3R v17.5 Weekly Analysis - LLM Insights
## Week 49: December 1-8, 2025

### 🎯 Strengths Identified
- **Endgame Performance**: 30% reduction in critical blunders in positions ≤6 pieces
- **Top1 Alignment**: Improved 3.3% from baseline, indicating better move selection
- **Mate Detection**: Faster recognition of mate threats (evident in reduced mate-in-2/3 misses)

### ⚠️ Weakness Themes

#### 1. Early Middlegame Pawn Structure (12 instances)
**FEN**: `r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq -`

**Issue**: Engine consistently plays Bc4 instead of d4, missing central control.

**Root Cause**: Piece-Square Tables (PST) may overvalue bishop activity on c4 diagonal in opening phase.

**Recommendation**: 
```python
# In v7p3r_evaluation.py
# Reduce bishop PST bonus for c4/f4 in opening (move 3-8)
if game_phase == 'opening' and move_number < 8:
    bishop_pst_bonus *= 0.85  # Reduce early diagonal development bonus
    center_pawn_bonus *= 1.15  # Increase d4/e4 pawn push value
```

#### 2. Queen Activation Timing (8 instances)
**Pattern**: Early queen development (moves 6-10) leading to tempo losses.

**Issue**: Queen comes out before minor pieces developed, gets harassed.

**Recommendation**:
- Add penalty for queen moves before ≥3 minor pieces developed
- Increase queen mobility value only after development threshold met

#### 3. Endgame King Activity (5 instances)
**Pattern**: King remains passive in K+P vs K+P endgames.

**Issue**: Not activating king quickly enough in simplified positions.

**Recommendation**:
```python
# In endgame evaluation
if total_pieces <= 6 and pawns_only:
    king_activity_bonus *= 1.5  # More aggressive king in pure pawn endgames
    # Add bonus for king centralization
    if abs(king_file - 4) <= 1 and abs(king_rank - 4) <= 1:
        score += 50  # Bonus for central king in endgame
```

### 📊 Code Review Priorities

**High Priority**:
1. **PST Tuning**: Reduce early bishop diagonal bonus (opening phase)
2. **Queen Development**: Add penalty for premature queen activity
3. **Endgame King**: Increase king activity weight in simplified positions

**Medium Priority**:
4. **Search Extensions**: Consider extending forced sequences in middlegame
5. **Time Management**: Some games flagging in bullet - review time allocation

**Low Priority**:
6. **Opening Book**: Consider expanding early responses (still repetitive)

### 📈 Version Comparison

v17.5 shows **significant improvement** over v17.1:
- Top1 Alignment: +3.3% (43.8% → 47.1%)
- Critical Blunders: -40% (7.0 → 4.2 per game)
- Average CPL: -156 (1498 → 1342)

**Endgame optimization is working!** The castling pruning and PST changes are having positive impact.

### 🎬 Next Steps for v17.6

1. Implement PST tuning for bishop/queen opening phase
2. Add endgame king activity boost
3. Test with 50+ games before deployment
4. Monitor early middlegame performance specifically

---
*Generated by LLM Analysis System*  
*Human review recommended before implementing changes*
```

---

## Implementation Files

### File 1: `llm_analyzer.py`

```python
"""
LLM-powered analysis of chess engine performance.
Generates actionable insights from technical metrics.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional
import os

logger = logging.getLogger(__name__)

# Try OpenAI first, fall back to local LLM
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available, will use local LLM only")


class LLMAnalyzer:
    """Generates AI-powered insights from chess analysis data."""
    
    def __init__(self, provider: str = "openai", api_key: str = None):
        """
        Initialize LLM analyzer.
        
        Args:
            provider: "openai" or "local"
            api_key: OpenAI API key (if using OpenAI)
        """
        self.provider = provider
        
        if provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ValueError("OpenAI library not installed")
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required")
            
            self.client = OpenAI(api_key=api_key)
            self.model = "gpt-4-turbo-preview"
        else:
            # Local LLM setup (Ollama)
            self.model = "llama3"
    
    def analyze_week(self, metrics_data: Dict, output_file: Path) -> bool:
        """
        Generate LLM analysis of weekly performance.
        
        Args:
            metrics_data: Dict with week's metrics and blunder patterns
            output_file: Path to save analysis markdown
            
        Returns:
            True if successful
        """
        try:
            # Generate prompt
            prompt = self._build_analysis_prompt(metrics_data)
            
            # Get LLM response
            if self.provider == "openai":
                analysis = self._analyze_with_openai(prompt)
            else:
                analysis = self._analyze_with_local(prompt)
            
            # Save to file
            with open(output_file, 'w') as f:
                f.write(analysis)
            
            logger.info(f"LLM analysis saved to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return False
    
    def _build_analysis_prompt(self, data: Dict) -> str:
        """Build analysis prompt from metrics."""
        
        json_data = json.dumps(data, indent=2)
        
        prompt = f"""You are a chess engine development advisor analyzing the performance of V7P3R chess engine.

Review the following weekly performance data and provide:

1. **Strengths Identified**: What is the engine doing well this week?
2. **Weakness Themes**: What patterns of mistakes appear repeatedly?
3. **Opening Issues**: Any specific opening lines causing problems?
4. **Middlegame Tactical Gaps**: Missed tactics or position evaluation errors?
5. **Endgame Problems**: Specific endgame types that need improvement?
6. **Code Review Suggestions**: Concrete recommendations for the next version:
   - Search improvements (depth, pruning, extensions)
   - Evaluation function tweaks (PST, material, mobility)
   - Time management adjustments
   - Bug fixes for specific positions

Format your response as a clear, actionable Markdown document with specific Python code suggestions where applicable.

Data:
{json_data}
"""
        return prompt
    
    def _analyze_with_openai(self, prompt: str) -> str:
        """Get analysis from OpenAI."""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert chess engine developer and analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    
    def _analyze_with_local(self, prompt: str) -> str:
        """Get analysis from local LLM (Ollama)."""
        import subprocess
        
        try:
            result = subprocess.run(
                ["ollama", "run", self.model, prompt],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                return result.stdout
            else:
                return f"# LLM Analysis Failed\n\nError: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return "# LLM Analysis Failed\n\nError: Timeout (120s exceeded)"
        except FileNotFoundError:
            return "# LLM Analysis Failed\n\nError: Ollama not installed. Install from https://ollama.ai"
```

### File 2: `docker-compose.yml`

```yaml
version: '3.8'

services:
  analytics:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      # Persistent report storage
      - ./analytics_reports:/workspace/reports:rw
      # Data directory for temp files
      - ./analytics_data:/workspace/data:rw
      # Environment variables
      - ./.env:/app/.env:ro
    environment:
      - STOCKFISH_PATH=/usr/local/bin/stockfish
      - WORK_DIR=/workspace
      - DAYS_BACK=7
      - WORKERS=12
      # LLM configuration
      - LLM_PROVIDER=${LLM_PROVIDER:-openai}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    container_name: v7p3r-analytics
    restart: "no"
    # Automatically remove container after completion
    command: >
      python weekly_pipeline_local.py
      --stockfish /usr/local/bin/stockfish
      --work-dir /workspace
      --days-back 7
      --workers 12
      --output-dir /workspace/reports
```

### File 3: `.env.example`

```bash
# LLM Provider: "openai" or "local"
LLM_PROVIDER=openai

# OpenAI API Key (if using OpenAI)
OPENAI_API_KEY=sk-your-key-here

# Ollama Model (if using local LLM)
OLLAMA_MODEL=llama3
```

---

## Windows Task Scheduler Setup

### Create Scheduled Task

**PowerShell Script**: `run_analytics.ps1`

```powershell
# V7P3R Weekly Analytics - Run with Docker Compose
# Schedule: Every Sunday at midnight

$ANALYTICS_DIR = "S:\Programming\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine\analytics"

Set-Location $ANALYTICS_DIR

Write-Host "Starting V7P3R Analytics..."
Write-Host "Time: $(Get-Date)"

# Run docker-compose
docker-compose up --build

# Check exit code
if ($LASTEXITCODE -eq 0) {
    Write-Host "Analytics completed successfully"
} else {
    Write-Host "Analytics failed with code $LASTEXITCODE"
}

# Cleanup
docker-compose down

Write-Host "Done at $(Get-Date)"
```

### Schedule Command

```powershell
# Run as Administrator in PowerShell

$Action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-File `"S:\Programming\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine\analytics\run_analytics.ps1`""

$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 12:00AM

$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable

Register-ScheduledTask -TaskName "V7P3R Weekly Analytics" `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "Run V7P3R chess engine analytics every Sunday midnight"
```

---

## Migration Path to Cloud

When ready to move to cloud:

1. **Same Docker image** - already cloud-ready
2. **Update volumes** - change to Cloud Storage mounts
3. **Add email** - enable SendGrid in pipeline
4. **Deploy** - `docker push` + Cloud Run job

**No code changes needed** - just configuration!

---

## Cost Comparison

### Local Setup
- **Hardware**: $0 (existing PC)
- **OpenAI LLM**: ~$0.05/week = $2.60/year
- **Total**: ~$2.60/year

### Cloud Setup (for reference)
- **Cloud Run**: $0.40/month
- **Cloud Storage**: $0.02/month
- **SendGrid**: $0/month (free tier)
- **OpenAI LLM**: $0.05/week
- **Total**: ~$8/year

**Savings**: $5.40/year + full control + easier debugging

---

## Next Steps

1. ✅ Shut down cloud infrastructure (DONE)
2. Create `llm_analyzer.py`
3. Create `weekly_pipeline_local.py` (enhanced version)
4. Create `docker-compose.yml`
5. Set up `.env` with OpenAI key
6. Test manual run
7. Schedule with Windows Task Scheduler
8. Monitor for 2-3 weeks
9. Review LLM insights
10. Iterate and improve!

---

**Status**: Ready to implement  
**Timeline**: 1-2 hours setup, 2-3 weeks testing  
**Risk**: Low (everything local, no cloud costs)
