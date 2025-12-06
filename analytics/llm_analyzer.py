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
                raise ValueError("OpenAI library not installed. Run: pip install openai")
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
            
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
            logger.info(f"Generating LLM analysis with {self.provider}...")
            
            # Generate prompt
            prompt = self._build_analysis_prompt(metrics_data)
            
            # Get LLM response
            if self.provider == "openai":
                analysis = self._analyze_with_openai(prompt)
            else:
                analysis = self._analyze_with_local(prompt)
            
            # Save to file
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(analysis)
            
            logger.info(f"✓ LLM analysis saved to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"✗ LLM analysis failed: {e}")
            # Save error report
            error_report = f"# LLM Analysis Failed\n\n**Error**: {str(e)}\n\n**Provider**: {self.provider}\n"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(error_report)
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

Format your response as a clear, actionable Markdown document with:
- Clear section headers (##)
- Bullet points for lists
- Code blocks with Python syntax where applicable
- Specific, implementable recommendations

Keep the response focused and actionable. Prioritize high-impact changes.

Data:
{json_data}
"""
        return prompt
    
    def _analyze_with_openai(self, prompt: str) -> str:
        """Get analysis from OpenAI."""
        
        logger.info(f"Calling OpenAI API ({self.model})...")
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert chess engine developer and analyst with deep knowledge of search algorithms, evaluation functions, and chess strategy."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2500
        )
        
        analysis = response.choices[0].message.content
        logger.info(f"✓ Received {len(analysis)} characters from OpenAI")
        
        return analysis
    
    def _analyze_with_local(self, prompt: str) -> str:
        """Get analysis from local LLM (Ollama)."""
        import subprocess
        
        logger.info(f"Calling local LLM ({self.model})...")
        
        try:
            result = subprocess.run(
                ["ollama", "run", self.model, prompt],
                capture_output=True,
                text=True,
                timeout=180  # 3 minutes
            )
            
            if result.returncode == 0:
                logger.info(f"✓ Received {len(result.stdout)} characters from local LLM")
                return result.stdout
            else:
                error_msg = f"# LLM Analysis Failed\n\n**Error**: {result.stderr}\n\n**Model**: {self.model}"
                return error_msg
                
        except subprocess.TimeoutExpired:
            return "# LLM Analysis Failed\n\n**Error**: Timeout (180s exceeded)\n\nTry reducing the data size or using a faster model."
        except FileNotFoundError:
            return "# LLM Analysis Failed\n\n**Error**: Ollama not installed.\n\nInstall from: https://ollama.ai\n\nThen run: `ollama pull llama3`"


def main():
    """Test LLM analyzer."""
    import argparse
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    parser = argparse.ArgumentParser(description="Test LLM Analyzer")
    parser.add_argument("--provider", default="openai", choices=["openai", "local"], help="LLM provider")
    parser.add_argument("--test-data", required=True, help="Path to JSON test data")
    parser.add_argument("--output", default="test_analysis.md", help="Output file")
    
    args = parser.parse_args()
    
    # Load test data
    with open(args.test_data, 'r') as f:
        data = json.load(f)
    
    # Run analysis
    analyzer = LLMAnalyzer(provider=args.provider)
    success = analyzer.analyze_week(data, Path(args.output))
    
    if success:
        print(f"\n✓ Analysis complete: {args.output}")
    else:
        print(f"\n✗ Analysis failed")


if __name__ == "__main__":
    main()
