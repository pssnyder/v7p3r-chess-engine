#!/usr/bin/env python3
"""
V7P3R Chess Engine Build Completeness Analysis
Analyzes all extracted beta candidate builds for:
- File completeness and dependencies
- Code complexity and architecture
- Independent runnability assessment
- Competitive potential evaluation
"""

import os
import json
import ast
import re
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

class ChessBuildAnalyzer:
    def __init__(self, builds_dir="builds_complete"):
        self.builds_dir = Path(builds_dir)
        self.analysis_results = {}
        self.core_modules = [
            'v7p3r_engine.py', 'v7p3r_search.py', 'v7p3r_scoring.py',
            'v7p3r_rules.py', 'v7p3r_game.py', 'v7p3r_config.py',
            'v7p3r_utils.py', 'v7p3r_pst.py', 'v7p3r_book.py',
            'v7p3r_ordering.py', 'v7p3r_time.py', 'v7p3r_move_ordering.py',
            'v7p3r_mvv_lva.py', 'v7p3r_primary_scoring.py', 
            'v7p3r_secondary_scoring.py', 'v7p3r_quiescence.py',
            'v7p3r_stockfish.py', 'v7p3r_tempo.py'
        ]
        
    def analyze_all_builds(self):
        print("🔍 ANALYZING ALL BETA CANDIDATE BUILDS")
        print("=" * 60)
        
        for build_dir in sorted(self.builds_dir.glob("v0.*_beta-candidate-*_COMPLETE")):
            if build_dir.is_dir():
                print(f"\n📦 Analyzing: {build_dir.name}")
                self.analyze_build(build_dir)
        
        self.generate_summary_report()
        
    def analyze_build(self, build_path):
        """Comprehensive analysis of a single build"""
        build_name = build_path.name
        
        # Load extraction info
        info_file = build_path / "EXTRACTION_INFO.json"
        extraction_info = {}
        if info_file.exists():
            try:
                with open(info_file, encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        extraction_info = json.loads(content)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Handle corrupted JSON files
                extraction_info = {'error': 'Could not parse extraction info'}
        
        analysis = {
            'build_name': build_name,
            'path': str(build_path),
            'extraction_info': extraction_info,
            'file_analysis': self.analyze_files(build_path),
            'dependency_analysis': self.check_dependencies(build_path),
            'code_metrics': self.calculate_code_metrics(build_path),
            'architecture_assessment': self.assess_architecture(build_path),
            'runnability_score': 0,
            'competitive_potential': 'UNKNOWN'
        }
        
        # Calculate runnability score
        analysis['runnability_score'] = self.calculate_runnability_score(analysis)
        
        # Assess competitive potential
        analysis['competitive_potential'] = self.assess_competitive_potential(analysis)
        
        self.analysis_results[build_name] = analysis
        
        # Print summary
        print(f"  Files: {analysis['file_analysis']['total_files']}")
        print(f"  Core modules: {analysis['dependency_analysis']['core_modules_present']}/{len(self.core_modules)}")
        print(f"  Python functions: {analysis['code_metrics']['total_functions']}")
        print(f"  Lines of code: {analysis['code_metrics']['total_lines']}")
        print(f"  Runnability: {analysis['runnability_score']}/100")
        print(f"  Competitive potential: {analysis['competitive_potential']}")
        
    def analyze_files(self, build_path):
        """Analyze file structure and types"""
        files = list(build_path.rglob("*"))
        file_types = Counter()
        
        for file_path in files:
            if file_path.is_file():
                ext = file_path.suffix.lower()
                file_types[ext] += 1
        
        return {
            'total_files': len([f for f in files if f.is_file()]),
            'file_types': dict(file_types),
            'has_main_files': any(f.name in ['play_chess.py', 'v7p3r_engine.py', 'main.py'] 
                                 for f in files),
            'has_config': any(f.suffix == '.json' for f in files),
            'has_tests': any('test' in f.name.lower() for f in files)
        }
        
    def check_dependencies(self, build_path):
        """Check for core engine dependencies"""
        python_files = list(build_path.rglob("*.py"))
        imports_found = set()
        core_modules_present = 0
        
        # Check which core modules are present
        for module in self.core_modules:
            if any(f.name == module for f in python_files):
                core_modules_present += 1
        
        # Analyze imports in Python files
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Find imports
                import_pattern = r'(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
                imports = re.findall(import_pattern, content)
                imports_found.update(imports)
                
            except Exception:
                continue
        
        return {
            'core_modules_present': core_modules_present,
            'total_core_modules': len(self.core_modules),
            'missing_core_modules': [m for m in self.core_modules 
                                   if not any(f.name == m for f in python_files)],
            'external_imports': [imp for imp in imports_found 
                               if not imp.startswith('v7p3r') and imp not in ['os', 'sys', 'json', 'time']],
            'completeness_ratio': core_modules_present / len(self.core_modules)
        }
        
    def calculate_code_metrics(self, build_path):
        """Calculate code complexity metrics"""
        python_files = list(build_path.rglob("*.py"))
        total_lines = 0
        total_functions = 0
        total_classes = 0
        complexity_score = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = len(content.splitlines())
                    total_lines += lines
                    
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            total_functions += 1
                        elif isinstance(node, ast.ClassDef):
                            total_classes += 1
                except:
                    # Fallback to regex counting
                    functions = len(re.findall(r'def\s+\w+', content))
                    classes = len(re.findall(r'class\s+\w+', content))
                    total_functions += functions
                    total_classes += classes
                    
            except Exception:
                continue
        
        # Calculate complexity score (simple heuristic)
        complexity_score = min(100, total_functions * 2 + total_classes * 5 + total_lines // 10)
        
        return {
            'total_lines': total_lines,
            'total_functions': total_functions,
            'total_classes': total_classes,
            'complexity_score': complexity_score,
            'python_files': len(python_files)
        }
        
    def assess_architecture(self, build_path):
        """Assess architectural maturity"""
        files = list(build_path.rglob("*.py"))
        
        # Check for architectural patterns
        has_engine_pattern = any('engine' in f.name.lower() for f in files)
        has_game_pattern = any('game' in f.name.lower() for f in files)
        has_search_pattern = any('search' in f.name.lower() for f in files)
        has_scoring_pattern = any('scor' in f.name.lower() for f in files)
        has_config_pattern = any('config' in f.name.lower() for f in files)
        
        modular_score = sum([
            has_engine_pattern, has_game_pattern, has_search_pattern,
            has_scoring_pattern, has_config_pattern
        ]) * 20  # Each pattern worth 20 points
        
        return {
            'has_engine_pattern': has_engine_pattern,
            'has_game_pattern': has_game_pattern,
            'has_search_pattern': has_search_pattern,
            'has_scoring_pattern': has_scoring_pattern,
            'has_config_pattern': has_config_pattern,
            'modular_score': modular_score,
            'architecture_level': 'ADVANCED' if modular_score >= 80 else 
                                'INTERMEDIATE' if modular_score >= 60 else 'BASIC'
        }
        
    def calculate_runnability_score(self, analysis):
        """Calculate how likely the build is to run independently"""
        score = 0
        
        # Core modules completeness (40 points)
        completeness = analysis['dependency_analysis']['completeness_ratio']
        score += completeness * 40
        
        # Has main entry point (20 points)
        if analysis['file_analysis']['has_main_files']:
            score += 20
            
        # Has configuration (10 points)
        if analysis['file_analysis']['has_config']:
            score += 10
            
        # Code complexity indicates maturity (15 points)
        complexity = min(15, analysis['code_metrics']['complexity_score'] / 10)
        score += complexity
        
        # Architecture quality (15 points)
        arch_score = analysis['architecture_assessment']['modular_score'] * 0.15
        score += arch_score
        
        return min(100, int(score))
        
    def assess_competitive_potential(self, analysis):
        """Assess competitive chess engine potential"""
        runnability = analysis['runnability_score']
        complexity = analysis['code_metrics']['complexity_score']
        completeness = analysis['dependency_analysis']['completeness_ratio']
        
        if runnability >= 80 and complexity >= 50 and completeness >= 0.8:
            return "HIGH"
        elif runnability >= 60 and complexity >= 30 and completeness >= 0.6:
            return "MEDIUM"
        elif runnability >= 40:
            return "LOW"
        else:
            return "EXPERIMENTAL"
            
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE BUILD ANALYSIS SUMMARY")
        print("=" * 80)
        
        # Sort builds by competitive potential and runnability
        sorted_builds = sorted(
            self.analysis_results.items(),
            key=lambda x: (x[1]['competitive_potential'], x[1]['runnability_score']),
            reverse=True
        )
        
        # Create summary table
        print("\n🏆 BUILD RANKINGS (Competitive Potential)")
        print("-" * 80)
        print(f"{'Rank':<4} {'Build Version':<35} {'Runnable':<8} {'Potential':<12} {'Files':<6}")
        print("-" * 80)
        
        for i, (build_name, analysis) in enumerate(sorted_builds, 1):
            version = build_name.split('_')[0]  # Extract version like v0.7.15
            runnability = f"{analysis['runnability_score']}/100"
            potential = analysis['competitive_potential']
            files = analysis['file_analysis']['total_files']
            
            print(f"{i:<4} {version:<35} {runnability:<8} {potential:<12} {files:<6}")
        
        # Top candidates for release
        print("\n🚀 TOP RELEASE CANDIDATES")
        print("-" * 50)
        top_candidates = [item for item in sorted_builds[:5] 
                         if item[1]['competitive_potential'] in ['HIGH', 'MEDIUM']]
        
        for build_name, analysis in top_candidates:
            version = build_name.split('_')[0]
            print(f"✅ {version}")
            print(f"   Runnability: {analysis['runnability_score']}/100")
            print(f"   Core modules: {analysis['dependency_analysis']['core_modules_present']}/{len(self.core_modules)}")
            print(f"   Functions: {analysis['code_metrics']['total_functions']}")
            print(f"   Architecture: {analysis['architecture_assessment']['architecture_level']}")
            print()
        
        # Save detailed JSON report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"build_analysis_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'summary': {
                    'total_builds_analyzed': len(self.analysis_results),
                    'high_potential_builds': len([a for a in self.analysis_results.values() 
                                                if a['competitive_potential'] == 'HIGH']),
                    'runnable_builds': len([a for a in self.analysis_results.values() 
                                          if a['runnability_score'] >= 70])
                },
                'detailed_analysis': self.analysis_results
            }, f, indent=2)
            
        print(f"📄 Detailed analysis saved to: {report_file}")
        print("\n✨ Analysis complete! Use top candidates for exe compilation and arena testing.")

if __name__ == "__main__":
    analyzer = ChessBuildAnalyzer()
    analyzer.analyze_all_builds()
