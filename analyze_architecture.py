# Architectural Deep Dive and Dependency Analysis
# Further analysis of engine architectures and missing dependencies

import os
import re
import json

def analyze_dependencies_and_errors(build_path):
    """Analyze missing dependencies and potential import errors"""
    analysis = {
        'missing_dependencies': [],
        'internal_imports': [],
        'external_imports': [],
        'import_errors': [],
        'architecture_details': {},
        'engine_readiness': 'unknown'
    }
    
    # Common chess engine dependencies
    standard_libs = {'os', 'sys', 'time', 'datetime', 'random', 're', 'json', 'logging', 'socket', 'io', 'hashlib', 'typing'}
    chess_libs = {'chess', 'pygame', 'yaml', 'numpy', 'tensorflow', 'torch'}
    
    for root, dirs, files in os.walk(build_path):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Find all imports
                    import_matches = re.findall(r'^(?:from\s+(\S+)\s+)?import\s+(.+)$', content, re.MULTILINE)
                    
                    for match in import_matches:
                        if match[0]:  # from X import Y
                            module = match[0].split('.')[0]
                        else:  # import X
                            module = match[1].split('.')[0].split(',')[0].strip()
                        
                        if module.startswith('v7p3r') or module in ['evaluation_engine', 'chess_game', 'viper']:
                            analysis['internal_imports'].append(module)
                        elif module in standard_libs:
                            continue  # Skip standard library
                        elif module in chess_libs:
                            analysis['external_imports'].append(module)
                        else:
                            analysis['external_imports'].append(module)
                
                except Exception as e:
                    analysis['import_errors'].append(f"{file}: {str(e)}")
    
    # Check for missing internal dependencies
    internal_modules = set(analysis['internal_imports'])
    existing_files = {os.path.splitext(f)[0] for f in os.listdir(build_path) if f.endswith('.py')}
    
    for module in internal_modules:
        if module not in existing_files:
            # Check if it exists in subdirectories
            found = False
            for root, dirs, files in os.walk(build_path):
                if f"{module}.py" in files:
                    found = True
                    break
            if not found:
                analysis['missing_dependencies'].append(module)
    
    return analysis

def classify_architecture_detailed(build_path, build_name):
    """Detailed architecture classification based on file structure and imports"""
    
    files = []
    for root, dirs, files_in_dir in os.walk(build_path):
        for file in files_in_dir:
            if file.endswith('.py'):
                rel_path = os.path.relpath(os.path.join(root, file), build_path)
                files.append(rel_path)
    
    architecture = {
        'type': 'unknown',
        'generation': 'unknown',
        'ui_type': 'none',
        'engine_pattern': 'unknown',
        'complexity_level': 'basic'
    }
    
    # Determine UI type
    if any('pygame' in f or 'gui' in f.lower() for f in files):
        architecture['ui_type'] = 'pygame_gui'
    elif any('streamlit' in f.lower() for f in files):
        architecture['ui_type'] = 'web_streamlit'
    elif any('webapp' in f.lower() or 'app.py' in f for f in files):
        architecture['ui_type'] = 'web_app'
    else:
        architecture['ui_type'] = 'headless'
    
    # Determine engine pattern
    if any('v7p3r_engine/' in f for f in files):
        architecture['engine_pattern'] = 'modular_v7p3r_dir'
    elif any(f.startswith('v7p3r_') for f in files):
        architecture['engine_pattern'] = 'modular_v7p3r_flat'
    elif any('evaluation_engine.py' in f for f in files):
        architecture['engine_pattern'] = 'evaluation_centric'
    elif any('chess_game.py' in f for f in files):
        architecture['engine_pattern'] = 'game_centric'
    elif any('viper' in f.lower() for f in files):
        architecture['engine_pattern'] = 'viper_based'
    
    # Determine generation based on patterns
    if 'v7p3r' in str(files):
        if any('v7p3r_engine/' in f for f in files):
            architecture['generation'] = 'v7p3r_gen2_modular'
        else:
            architecture['generation'] = 'v7p3r_gen3_flat'
    elif 'viper' in str(files).lower():
        architecture['generation'] = 'viper_gen1'
    elif 'evaluation_engine' in str(files):
        architecture['generation'] = 'eval_engine_gen1'
    else:
        architecture['generation'] = 'early_prototype'
    
    # Determine complexity level
    total_files = len(files)
    if total_files >= 10:
        architecture['complexity_level'] = 'advanced'
    elif total_files >= 5:
        architecture['complexity_level'] = 'intermediate'
    else:
        architecture['complexity_level'] = 'basic'
    
    return architecture

def generate_architecture_report(builds_dir):
    """Generate detailed architecture and dependency analysis"""
    
    all_builds = []
    
    print("Performing detailed architecture analysis...")
    
    for build_name in sorted(os.listdir(builds_dir)):
        build_path = os.path.join(builds_dir, build_name)
        if os.path.isdir(build_path) and not build_name.endswith('.md'):
            print(f"  Analyzing {build_name}...")
            
            # Get dependency analysis
            dep_analysis = analyze_dependencies_and_errors(build_path)
            
            # Get detailed architecture
            arch_analysis = classify_architecture_detailed(build_path, build_name)
            
            build_info = {
                'name': build_name,
                'dependencies': dep_analysis,
                'architecture': arch_analysis
            }
            
            all_builds.append(build_info)
    
    # Generate report
    report = """# Detailed Architecture & Dependency Analysis
## Generated: July 24, 2025

### Architecture Evolution Overview

This analysis reveals the architectural evolution of the V7P3R chess engine through distinct generations and design patterns.

"""
    
    # Architecture classification table
    report += "## Architecture Classification Matrix\n\n"
    report += "| Build | Generation | Engine Pattern | UI Type | Complexity | Missing Deps |\n"
    report += "|-------|------------|----------------|---------|------------|---------------|\n"
    
    for build in all_builds:
        arch = build['architecture']
        deps = build['dependencies']
        missing_count = len(deps['missing_dependencies'])
        
        report += f"| {build['name']} | {arch['generation']} | {arch['engine_pattern']} | {arch['ui_type']} | {arch['complexity_level']} | {missing_count} |\n"
    
    # Group by generation
    report += "\n## Architectural Generations\n\n"
    
    generations = {}
    for build in all_builds:
        gen = build['architecture']['generation']
        if gen not in generations:
            generations[gen] = []
        generations[gen].append(build)
    
    for gen, builds in generations.items():
        report += f"### {gen.replace('_', ' ').title()}\n"
        report += f"**Count**: {len(builds)} builds\n\n"
        
        for build in builds:
            arch = build['architecture']
            report += f"- **{build['name']}**: {arch['engine_pattern']} with {arch['ui_type']} interface ({arch['complexity_level']} complexity)\n"
        
        # Common characteristics
        ui_types = [b['architecture']['ui_type'] for b in builds]
        engine_patterns = [b['architecture']['engine_pattern'] for b in builds]
        
        report += f"\n**Common UI**: {max(set(ui_types), key=ui_types.count) if ui_types else 'none'}\n"
        report += f"**Common Pattern**: {max(set(engine_patterns), key=engine_patterns.count) if engine_patterns else 'none'}\n\n"
    
    # Dependency analysis
    report += "## Dependency Health Analysis\n\n"
    
    # Builds with missing dependencies
    problematic_builds = [b for b in all_builds if b['dependencies']['missing_dependencies']]
    clean_builds = [b for b in all_builds if not b['dependencies']['missing_dependencies']]
    
    report += f"### ✅ Clean Builds (No Missing Dependencies): {len(clean_builds)}\n"
    for build in clean_builds[:5]:  # Show top 5
        report += f"- {build['name']}\n"
    if len(clean_builds) > 5:
        report += f"- ... and {len(clean_builds) - 5} more\n"
    
    report += f"\n### ⚠️ Builds with Missing Dependencies: {len(problematic_builds)}\n"
    for build in problematic_builds:
        missing = build['dependencies']['missing_dependencies']
        report += f"- **{build['name']}**: Missing {', '.join(missing)}\n"
    
    # Unique architecture candidates
    report += "\n## Unique Architecture Candidates\n\n"
    report += "Based on this analysis, here are the most representative builds for each architectural approach:\n\n"
    
    # Find best representative for each pattern
    patterns = {}
    for build in all_builds:
        pattern = build['architecture']['engine_pattern']
        if pattern not in patterns:
            patterns[pattern] = []
        patterns[pattern].append(build)
    
    for pattern, builds in patterns.items():
        # Find the most complete build in this pattern
        best_build = min(builds, key=lambda x: len(x['dependencies']['missing_dependencies']))
        complexity_order = {'advanced': 3, 'intermediate': 2, 'basic': 1}
        if len([b for b in builds if len(b['dependencies']['missing_dependencies']) == 0]) > 0:
            # Among clean builds, pick the most complex
            clean_builds_in_pattern = [b for b in builds if len(b['dependencies']['missing_dependencies']) == 0]
            best_build = max(clean_builds_in_pattern, key=lambda x: complexity_order.get(x['architecture']['complexity_level'], 0))
        
        report += f"### {pattern.replace('_', ' ').title()}\n"
        report += f"**Best Representative**: {best_build['name']}\n"
        report += f"- Generation: {best_build['architecture']['generation']}\n"
        report += f"- UI Type: {best_build['architecture']['ui_type']}\n"
        report += f"- Complexity: {best_build['architecture']['complexity_level']}\n"
        report += f"- Missing Dependencies: {len(best_build['dependencies']['missing_dependencies'])}\n\n"
    
    # Recommendations for testing priority
    report += "## Testing Priority Recommendations\n\n"
    
    # Priority 1: Clean, advanced builds
    priority_1 = [b for b in all_builds if not b['dependencies']['missing_dependencies'] and b['architecture']['complexity_level'] == 'advanced']
    priority_1.sort(key=lambda x: x['name'])
    
    report += "### Priority 1: Clean Advanced Builds (Ready for Immediate Testing)\n"
    for build in priority_1:
        report += f"- **{build['name']}** ({build['architecture']['generation']}, {build['architecture']['engine_pattern']})\n"
    
    # Priority 2: Clean intermediate builds
    priority_2 = [b for b in all_builds if not b['dependencies']['missing_dependencies'] and b['architecture']['complexity_level'] == 'intermediate']
    priority_2.sort(key=lambda x: x['name'])
    
    report += "\n### Priority 2: Clean Intermediate Builds\n"
    for build in priority_2:
        report += f"- **{build['name']}** ({build['architecture']['generation']}, {build['architecture']['engine_pattern']})\n"
    
    # Priority 3: Unique architectures worth fixing
    unique_patterns = set(b['architecture']['engine_pattern'] for b in all_builds)
    priority_3 = []
    for pattern in unique_patterns:
        pattern_builds = [b for b in all_builds if b['architecture']['engine_pattern'] == pattern]
        # Find the best build in this pattern
        best_in_pattern = min(pattern_builds, key=lambda x: len(x['dependencies']['missing_dependencies']))
        if len(best_in_pattern['dependencies']['missing_dependencies']) > 0:
            priority_3.append(best_in_pattern)
    
    report += "\n### Priority 3: Unique Architectures Worth Fixing\n"
    for build in priority_3:
        missing = build['dependencies']['missing_dependencies']
        report += f"- **{build['name']}** (Missing: {', '.join(missing)})\n"
    
    # Write the report
    with open(os.path.join(builds_dir, "ARCHITECTURE_ANALYSIS_REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report)
    
    print("Architecture analysis complete!")
    print(f"Report saved to: {builds_dir}/ARCHITECTURE_ANALYSIS_REPORT.md")

if __name__ == "__main__":
    generate_architecture_report("builds")
