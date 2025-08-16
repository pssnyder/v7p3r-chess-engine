# Comprehensive Build Analysis Script
# Analyzes code completeness, complexity, architecture, and competitive potential

import os
import re
import json
from collections import defaultdict, Counter

def analyze_python_file(filepath):
    """Analyze a Python file for imports, functions, classes, and complexity indicators"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except:
        return None
    
    analysis = {
        'imports': [],
        'functions': [],
        'classes': [],
        'lines': len(content.split('\n')),
        'complexity_indicators': {
            'if_statements': len(re.findall(r'\bif\b', content)),
            'for_loops': len(re.findall(r'\bfor\b', content)),
            'while_loops': len(re.findall(r'\bwhile\b', content)),
            'try_except': len(re.findall(r'\btry:\b', content)),
            'nested_levels': 0  # Simplified - would need proper AST for accurate count
        }
    }
    
    # Extract imports
    import_matches = re.findall(r'^(?:from\s+(\S+)\s+)?import\s+(.+)$', content, re.MULTILINE)
    for match in import_matches:
        if match[0]:  # from X import Y
            analysis['imports'].append(f"from {match[0]} import {match[1].strip()}")
        else:  # import X
            analysis['imports'].append(f"import {match[1].strip()}")
    
    # Extract function definitions
    func_matches = re.findall(r'def\s+(\w+)\s*\([^)]*\):', content)
    analysis['functions'] = func_matches
    
    # Extract class definitions
    class_matches = re.findall(r'class\s+(\w+)(?:\([^)]*\))?:', content)
    analysis['classes'] = class_matches
    
    return analysis

def analyze_build(build_path):
    """Comprehensive analysis of a single build"""
    build_name = os.path.basename(build_path)
    
    analysis = {
        'build_name': build_name,
        'file_stats': {'total': 0, 'python': 0, 'config': 0, 'db': 0, 'docs': 0},
        'python_files': {},
        'imports': set(),
        'missing_imports': set(),
        'functions': [],
        'classes': [],
        'architecture_type': 'unknown',
        'completeness_score': 0.0,
        'complexity_score': 0.0,
        'competitive_potential': 'unknown',
        'unique_features': [],
        'engine_components': {}
    }
    
    # Analyze all files
    for root, dirs, files in os.walk(build_path):
        for file in files:
            filepath = os.path.join(root, file)
            rel_path = os.path.relpath(filepath, build_path)
            
            analysis['file_stats']['total'] += 1
            
            if file.endswith('.py'):
                analysis['file_stats']['python'] += 1
                py_analysis = analyze_python_file(filepath)
                if py_analysis:
                    analysis['python_files'][rel_path] = py_analysis
                    analysis['imports'].update(py_analysis['imports'])
                    analysis['functions'].extend(py_analysis['functions'])
                    analysis['classes'].extend(py_analysis['classes'])
            
            elif file.endswith(('.json', '.yaml', '.yml')):
                analysis['file_stats']['config'] += 1
            elif file.endswith('.db'):
                analysis['file_stats']['db'] += 1
            elif file.endswith('.md'):
                analysis['file_stats']['docs'] += 1
    
    # Determine architecture type
    if any('pygame' in imp for imp in analysis['imports']):
        analysis['architecture_type'] = 'GUI-based'
    elif any('v7p3r_engine' in file for file in analysis['python_files']):
        analysis['architecture_type'] = 'Modular v7p3r'
    elif any('evaluation_engine' in file for file in analysis['python_files']):
        analysis['architecture_type'] = 'Evaluation-centric'
    elif any('chess_game' in file for file in analysis['python_files']):
        analysis['architecture_type'] = 'Game-centric'
    elif any('v7p3r' in file.lower() for file in analysis['python_files']):
        analysis['architecture_type'] = 'V7P3R-based'
    
    # Calculate complexity score (relative)
    total_functions = len(analysis['functions'])
    total_classes = len(analysis['classes'])
    total_imports = len(analysis['imports'])
    total_files = analysis['file_stats']['python']
    
    complexity_raw = (total_functions * 1.0 + 
                     total_classes * 2.0 + 
                     total_imports * 0.5 + 
                     total_files * 1.5)
    
    analysis['complexity_raw'] = complexity_raw
    
    # Check for specific engine components
    components = {
        'search_algorithm': any('search' in func.lower() for func in analysis['functions']),
        'evaluation_function': any('evaluat' in func.lower() for func in analysis['functions']),
        'move_ordering': any('order' in func.lower() for func in analysis['functions']),
        'opening_book': any('opening' in imp.lower() or 'book' in imp.lower() for imp in analysis['imports']),
        'endgame_tables': any('endgame' in func.lower() for func in analysis['functions']),
        'time_management': any('time' in func.lower() for func in analysis['functions']),
        'transposition_table': any('transposition' in func.lower() or 'hash' in func.lower() for func in analysis['functions']),
        'quiescence_search': any('quiescence' in func.lower() for func in analysis['functions']),
        'pruning': any('prune' in func.lower() or 'alpha' in func.lower() for func in analysis['functions'])
    }
    
    analysis['engine_components'] = components
    
    # Calculate completeness score
    completeness = sum(components.values()) / len(components)
    analysis['completeness_score'] = completeness
    
    return analysis

def main():
    builds_dir = "builds"
    all_analyses = []
    
    print("=== COMPREHENSIVE BUILD ANALYSIS ===")
    print("Analyzing all beta candidates for completeness, complexity, and architecture...")
    
    # Analyze each build
    for build_name in sorted(os.listdir(builds_dir)):
        build_path = os.path.join(builds_dir, build_name)
        if os.path.isdir(build_path):
            print(f"Analyzing {build_name}...")
            analysis = analyze_build(build_path)
            all_analyses.append(analysis)
    
    # Normalize complexity scores (0.01 to 1.0)
    max_complexity = max(a['complexity_raw'] for a in all_analyses)
    min_complexity = min(a['complexity_raw'] for a in all_analyses)
    
    for analysis in all_analyses:
        if max_complexity > min_complexity:
            normalized = 0.01 + 0.99 * (analysis['complexity_raw'] - min_complexity) / (max_complexity - min_complexity)
        else:
            normalized = 0.5
        analysis['complexity_score'] = round(normalized, 3)
    
    # Determine competitive potential
    for analysis in all_analyses:
        score = 0
        if analysis['completeness_score'] > 0.7: score += 3
        elif analysis['completeness_score'] > 0.4: score += 2
        else: score += 1
        
        if analysis['file_stats']['python'] > 5: score += 2
        if analysis['architecture_type'] in ['Modular v7p3r', 'Evaluation-centric']: score += 2
        if any('stockfish' in imp.lower() for imp in analysis['imports']): score += 1
        
        if score >= 7: analysis['competitive_potential'] = 'High'
        elif score >= 4: analysis['competitive_potential'] = 'Medium'  
        else: analysis['competitive_potential'] = 'Low'
    
    # Generate comprehensive report
    generate_comprehensive_report(all_analyses)
    
    return all_analyses

def generate_comprehensive_report(analyses):
    """Generate detailed analysis report"""
    
    report = """# Comprehensive Beta Candidate Analysis Report
## Generated: July 24, 2025

### Executive Summary
Analysis of 17 beta candidates reveals distinct architectural evolution patterns and varying levels of competitive potential.

"""
    
    # Overview table
    report += "## Build Overview & Competitive Assessment\n\n"
    report += "| Build | Architecture | Completeness | Complexity | Competitive Potential | Python Files | Functions | Classes |\n"
    report += "|-------|--------------|--------------|------------|---------------------|--------------|-----------|----------|\n"
    
    for analysis in sorted(analyses, key=lambda x: x['complexity_score'], reverse=True):
        report += f"| {analysis['build_name']} | {analysis['architecture_type']} | {analysis['completeness_score']:.2f} | {analysis['complexity_score']:.3f} | {analysis['competitive_potential']} | {analysis['file_stats']['python']} | {len(analysis['functions'])} | {len(analysis['classes'])} |\n"
    
    # Architecture classification
    report += "\n## Architecture Classification\n\n"
    arch_groups = defaultdict(list)
    for analysis in analyses:
        arch_groups[analysis['architecture_type']].append(analysis['build_name'])
    
    for arch_type, builds in arch_groups.items():
        report += f"### {arch_type}\n"
        for build in builds:
            report += f"- {build}\n"
        report += "\n"
    
    # Competitive potential grouping
    report += "## Competitive Potential Rankings\n\n"
    potential_groups = defaultdict(list)
    for analysis in analyses:
        potential_groups[analysis['competitive_potential']].append(analysis)
    
    for potential in ['High', 'Medium', 'Low']:
        if potential in potential_groups:
            report += f"### {potential} Competitive Potential\n"
            for analysis in sorted(potential_groups[potential], key=lambda x: x['complexity_score'], reverse=True):
                report += f"- **{analysis['build_name']}** (Complexity: {analysis['complexity_score']:.3f}, Completeness: {analysis['completeness_score']:.2f})\n"
            report += "\n"
    
    # Engine components analysis
    report += "## Engine Components Comparison\n\n"
    report += "| Build | Search | Evaluation | Move Ordering | Opening Book | Time Mgmt | Transposition | Quiescence | Pruning |\n"
    report += "|-------|--------|------------|---------------|--------------|-----------|---------------|------------|----------|\n"
    
    for analysis in sorted(analyses, key=lambda x: x['build_name']):
        components = analysis['engine_components']
        row = f"| {analysis['build_name']} |"
        for comp in ['search_algorithm', 'evaluation_function', 'move_ordering', 'opening_book', 'time_management', 'transposition_table', 'quiescence_search', 'pruning']:
            row += " ✓ |" if components[comp] else " ✗ |"
        report += row + "\n"
    
    # Detailed individual analysis
    report += "\n## Detailed Build Analysis\n\n"
    
    for analysis in sorted(analyses, key=lambda x: x['complexity_score'], reverse=True):
        report += f"### {analysis['build_name']}\n"
        report += f"- **Architecture**: {analysis['architecture_type']}\n"
        report += f"- **Completeness Score**: {analysis['completeness_score']:.2f}/1.0\n"
        report += f"- **Complexity Score**: {analysis['complexity_score']:.3f}/1.0\n"
        report += f"- **Competitive Potential**: {analysis['competitive_potential']}\n"
        report += f"- **Files**: {analysis['file_stats']['python']} Python, {analysis['file_stats']['config']} Config, {analysis['file_stats']['db']} DB\n"
        report += f"- **Code Stats**: {len(analysis['functions'])} functions, {len(analysis['classes'])} classes\n"
        
        # Key features
        key_features = []
        if analysis['engine_components']['search_algorithm']: key_features.append("Search Algorithm")
        if analysis['engine_components']['evaluation_function']: key_features.append("Evaluation Function")  
        if analysis['engine_components']['opening_book']: key_features.append("Opening Book")
        if analysis['engine_components']['time_management']: key_features.append("Time Management")
        
        if key_features:
            report += f"- **Key Features**: {', '.join(key_features)}\n"
        
        # Architecture insights
        if 'pygame' in str(analysis['imports']):
            report += f"- **Note**: GUI-enabled version with visual interface\n"
        if any('stockfish' in imp.lower() for imp in analysis['imports']):
            report += f"- **Note**: Includes Stockfish integration for testing\n"
        if any('genetic' in imp.lower() or 'ga_' in imp.lower() for imp in analysis['imports']):
            report += f"- **Note**: Includes genetic algorithm components\n"
            
        report += "\n"
    
    # Recommendations
    report += "## Recommendations\n\n"
    report += "### Immediate Testing Priority (High Competitive Potential)\n"
    high_potential = [a for a in analyses if a['competitive_potential'] == 'High']
    for analysis in sorted(high_potential, key=lambda x: x['complexity_score'], reverse=True):
        report += f"1. **{analysis['build_name']}** - {analysis['architecture_type']} architecture with {analysis['completeness_score']:.2f} completeness\n"
    
    report += "\n### Unique Architecture Candidates\n"
    unique_archs = set(a['architecture_type'] for a in analyses)
    for arch in unique_archs:
        best_in_arch = max([a for a in analyses if a['architecture_type'] == arch], key=lambda x: x['completeness_score'])
        report += f"- **{arch}**: {best_in_arch['build_name']} (best in category)\n"
    
    report += "\n### Code Mining Opportunities\n"
    report += "Builds with unique features worth extracting:\n"
    for analysis in analyses:
        unique_features = []
        if analysis['engine_components']['quiescence_search']: unique_features.append("Quiescence Search")
        if analysis['engine_components']['transposition_table']: unique_features.append("Transposition Tables")
        if any('genetic' in imp.lower() for imp in analysis['imports']): unique_features.append("Genetic Algorithm")
        if any('reinforcement' in imp.lower() or '_rl_' in imp.lower() for imp in analysis['imports']): unique_features.append("Reinforcement Learning")
        
        if unique_features:
            report += f"- **{analysis['build_name']}**: {', '.join(unique_features)}\n"
    
    # Write report
    with open("builds/COMPREHENSIVE_ANALYSIS_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("Comprehensive analysis complete!")
    print("Report saved to: builds/COMPREHENSIVE_ANALYSIS_REPORT.md")

if __name__ == "__main__":
    main()
