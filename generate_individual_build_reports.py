#!/usr/bin/env python3
"""
Individual Build Analysis and Report Generator
Generates comprehensive reports for each beta candidate build
"""

import os
import json
import ast
import re
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import hashlib

class IndividualBuildAnalyzer:
    def __init__(self, builds_dir="builds"):
        self.builds_dir = Path(builds_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Known file patterns for chess engines
        self.critical_files = {
            'main_entry': ['main.py', 'play_chess.py', 'chess_game.py', 'v7p3r.py', 'engine.py', 'viper.py', '__main__.py'],
            'engine_core': ['engine.py', 'v7p3r_engine.py', 'chess_engine.py', 'evaluation_engine.py'],
            'game_logic': ['chess_game.py', 'v7p3r_game.py', 'game.py', 'rules.py', 'v7p3r_rules.py'],
            'search': ['search.py', 'v7p3r_search.py', 'minimax.py', 'alphabeta.py'],
            'evaluation': ['eval.py', 'evaluation.py', 'v7p3r_scoring.py', 'scoring.py'],
            'config': ['config.json', 'config.yaml', 'config.yml', 'settings.json']
        }
        
        # Common issues to check for
        self.issue_patterns = {
            'syntax_errors': [r'def\s+\w+\([^)]*\)\s*$', r'class\s+\w+.*:\s*$'],
            'incomplete_functions': [r'def\s+\w+.*:\s*pass\s*$', r'def\s+\w+.*:\s*\.\.\.'],
            'debug_prints': [r'print\s*\(.*debug.*\)', r'print\s*\(.*DEBUG.*\)', r'print\s*\(.*test.*\)'],
            'hardcoded_paths': [r'["\'][C-Z]:\\', r'["\'][C-Z]:/'],
            'todo_comments': [r'#\s*TODO', r'#\s*FIXME', r'#\s*BUG', r'#\s*HACK']
        }

    def analyze_all_builds(self):
        """Analyze all builds and generate individual reports"""
        print("🔍 GENERATING INDIVIDUAL BUILD REPORTS")
        print("=" * 60)
        
        if not self.builds_dir.exists():
            print(f"❌ Builds directory not found: {self.builds_dir}")
            return
            
        # Find all build directories - more flexible pattern matching
        build_dirs = []
        for d in self.builds_dir.iterdir():
            if d.is_dir():
                # Match various patterns: v0.*, beta-candidate*, backup*, etc.
                name_lower = d.name.lower()
                if any(pattern in name_lower for pattern in ['v0.', 'beta', 'candidate', 'backup', 'build']):
                    build_dirs.append(d)
        
        build_dirs = sorted(build_dirs, key=lambda x: x.name)
        
        if not build_dirs:
            print("❌ No build directories found with recognizable patterns")
            print("    Looking for directories containing: v0., beta, candidate, backup, build")
            return
        
        print(f"Found {len(build_dirs)} build directories:")
        for build_dir in build_dirs:
            print(f"  - {build_dir.name}")
        print()
        
        for build_dir in build_dirs:
            print(f"📦 Analyzing: {build_dir.name}")
            self.analyze_build(build_dir)
            
        print(f"\n✨ All individual reports generated!")

    def analyze_build(self, build_path):
        """Perform comprehensive analysis of a single build"""
        build_name = build_path.name
        
        analysis = {
            'build_info': self.get_build_info(build_path),
            'file_analysis': self.analyze_files(build_path),
            'folder_structure': self.analyze_folder_structure(build_path),
            'python_analysis': self.analyze_python_files(build_path),
            'dependencies': self.analyze_dependencies(build_path),
            'critical_files': self.check_critical_files(build_path),
            'potential_issues': self.identify_issues(build_path),
            'size_analysis': self.analyze_sizes(build_path),
            'completeness_score': 0.0
        }
        
        # Calculate completeness score
        analysis['completeness_score'] = self.calculate_completeness(analysis)
        
        # Generate the report
        self.generate_build_report(build_path, analysis)
        
        print(f"  ✅ Report generated: {build_name}/BUILD_ANALYSIS_REPORT.md")

    def get_build_info(self, build_path):
        """Extract build information"""
        info = {
            'name': build_path.name,
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_files': 0,
            'total_size_mb': 0.0,
            'version': self.extract_version(build_path.name),
            'tag': self.extract_tag(build_path.name)
        }
        
        # Load extraction info if available
        extraction_info_file = build_path / "EXTRACTION_INFO.json"
        if extraction_info_file.exists():
            try:
                with open(extraction_info_file, 'r', encoding='utf-8') as f:
                    extraction_data = json.load(f)
                    info['extraction_date'] = extraction_data.get('date', 'unknown')
                    info['files_extracted'] = extraction_data.get('filesExtracted', 0)
            except:
                pass
        
        return info

    def extract_version(self, build_name):
        """Extract version from build name - more flexible patterns"""
        # Try various version patterns
        patterns = [
            r'(v\d+\.\d+\.\d+)',           # v0.1.2
            r'(v\d+\.\d+)',                # v0.1
            r'(\d+\.\d+\.\d+)',            # 0.1.2
            r'(\d+\.\d+)',                 # 0.1
            r'(beta-candidate-\d+)',       # beta-candidate-1
            r'(backup-\d+)',               # backup-1
            r'(build-\d+)'                 # build-1
        ]
        
        for pattern in patterns:
            match = re.search(pattern, build_name.lower())
            if match:
                return match.group(1)
        
        return build_name  # Return full name if no pattern matches

    def extract_tag(self, build_name):
        """Extract tag from build name - more flexible patterns"""
        # Try to find various tag patterns
        patterns = [
            r'(beta-candidate-\d+)',
            r'(alpha-\d+)',
            r'(release-\d+)',
            r'(backup-\d+)',
            r'(snapshot-\d+)',
            r'(build-\d+)',
            r'(test-\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, build_name.lower())
            if match:
                return match.group(1)
        
        # If no specific tag found, try to extract meaningful parts
        parts = build_name.split('_')
        if len(parts) > 1:
            return '_'.join(parts[1:])
        
        return 'unknown'

    def analyze_files(self, build_path):
        """Analyze all files in the build"""
        file_analysis = {
            'by_extension': defaultdict(lambda: {'count': 0, 'total_size': 0, 'files': []}),
            'total_count': 0,
            'total_size': 0,
            'largest_files': [],
            'empty_files': [],
            'duplicate_files': defaultdict(list)
        }
        
        file_hashes = {}
        all_files = []
        
        for file_path in build_path.rglob('*'):
            if file_path.is_file() and file_path.name != 'BUILD_ANALYSIS_REPORT.md':
                all_files.append(file_path)
                
                file_size = file_path.stat().st_size
                file_ext = file_path.suffix.lower() or '.no_extension'
                relative_path = file_path.relative_to(build_path)
                
                file_analysis['total_count'] += 1
                file_analysis['total_size'] += file_size
                
                # By extension
                file_analysis['by_extension'][file_ext]['count'] += 1
                file_analysis['by_extension'][file_ext]['total_size'] += file_size
                file_analysis['by_extension'][file_ext]['files'].append({
                    'name': file_path.name,
                    'path': str(relative_path),
                    'size': file_size
                })
                
                # Empty files
                if file_size == 0:
                    file_analysis['empty_files'].append(str(relative_path))
                
                # Calculate hash for duplicate detection
                try:
                    with open(file_path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                        if file_hash in file_hashes:
                            file_analysis['duplicate_files'][file_hash].append(str(relative_path))
                        else:
                            file_hashes[file_hash] = str(relative_path)
                            file_analysis['duplicate_files'][file_hash] = [str(relative_path)]
                except:
                    pass
        
        # Remove non-duplicates
        file_analysis['duplicate_files'] = {k: v for k, v in file_analysis['duplicate_files'].items() if len(v) > 1}
        
        # Largest files
        all_files.sort(key=lambda x: x.stat().st_size, reverse=True)
        file_analysis['largest_files'] = [
            {
                'name': f.name,
                'path': str(f.relative_to(build_path)),
                'size_mb': round(f.stat().st_size / (1024 * 1024), 3)
            }
            for f in all_files[:10]
        ]
        
        return file_analysis

    def analyze_folder_structure(self, build_path):
        """Analyze folder structure and organization"""
        structure = {
            'folders': [],
            'depth': 0,
            'organization_score': 0.0,
            'common_patterns': []
        }
        
        max_depth = 0
        folder_info = []
        
        for folder_path in build_path.rglob('*'):
            if folder_path.is_dir():
                relative_path = folder_path.relative_to(build_path)
                depth = len(relative_path.parts)
                max_depth = max(max_depth, depth)
                
                file_count = len([f for f in folder_path.iterdir() if f.is_file()])
                subfolder_count = len([f for f in folder_path.iterdir() if f.is_dir()])
                
                folder_info.append({
                    'name': folder_path.name,
                    'path': str(relative_path),
                    'depth': depth,
                    'file_count': file_count,
                    'subfolder_count': subfolder_count
                })
        
        structure['folders'] = sorted(folder_info, key=lambda x: x['path'])
        structure['depth'] = max_depth
        
        # Identify common patterns
        folder_names = [f['name'] for f in folder_info]
        common_folders = ['docs', 'test', 'tests', 'testing', 'config', 'data', 'images', 'temp', '__pycache__']
        structure['common_patterns'] = [name for name in common_folders if name in folder_names]
        
        # Calculate organization score
        has_docs = any('doc' in name.lower() for name in folder_names)
        has_tests = any('test' in name.lower() for name in folder_names)
        has_config = any('config' in name.lower() for name in folder_names)
        reasonable_depth = max_depth <= 4
        
        organization_score = sum([has_docs, has_tests, has_config, reasonable_depth]) / 4.0
        structure['organization_score'] = organization_score
        
        return structure

    def analyze_python_files(self, build_path):
        """Analyze Python files for syntax, structure, and quality"""
        python_analysis = {
            'files': [],
            'total_lines': 0,
            'total_functions': 0,
            'total_classes': 0,
            'syntax_errors': [],
            'import_errors': [],
            'complexity_issues': [],
            'code_quality_score': 0.0
        }
        
        python_files = list(build_path.rglob('*.py'))
        
        for py_file in python_files:
            file_info = self.analyze_python_file(py_file, build_path)
            python_analysis['files'].append(file_info)
            
            python_analysis['total_lines'] += file_info['lines']
            python_analysis['total_functions'] += file_info['functions']
            python_analysis['total_classes'] += file_info['classes']
            
            if file_info['syntax_errors']:
                python_analysis['syntax_errors'].extend(file_info['syntax_errors'])
            
            if file_info['import_errors']:
                python_analysis['import_errors'].extend(file_info['import_errors'])
            
            if file_info['complexity_score'] < 0.5:
                python_analysis['complexity_issues'].append(file_info['relative_path'])
        
        # Calculate overall code quality score
        if python_files:
            avg_complexity = sum(f['complexity_score'] for f in python_analysis['files']) / len(python_files)
            syntax_penalty = len(python_analysis['syntax_errors']) * 0.1
            import_penalty = len(python_analysis['import_errors']) * 0.05
            
            python_analysis['code_quality_score'] = max(0.0, avg_complexity - syntax_penalty - import_penalty)
        
        return python_analysis

    def analyze_python_file(self, file_path, build_path):
        """Analyze individual Python file"""
        relative_path = file_path.relative_to(build_path)
        
        file_info = {
            'name': file_path.name,
            'relative_path': str(relative_path),
            'size': file_path.stat().st_size,
            'lines': 0,
            'functions': 0,
            'classes': 0,
            'imports': [],
            'syntax_errors': [],
            'import_errors': [],
            'complexity_score': 0.0,
            'issues': []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Count lines (non-empty, non-comment)
            lines = content.split('\n')
            code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
            file_info['lines'] = len(code_lines)
            
            # Parse AST for structure analysis
            try:
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        file_info['functions'] += 1
                    elif isinstance(node, ast.ClassDef):
                        file_info['classes'] += 1
                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                file_info['imports'].append(alias.name)
                        elif isinstance(node, ast.ImportFrom) and node.module:
                            file_info['imports'].append(node.module)
                
            except SyntaxError as e:
                file_info['syntax_errors'].append(f"{relative_path}:{e.lineno}: {e.msg}")
            except Exception as e:
                file_info['syntax_errors'].append(f"{relative_path}: Parse error - {str(e)}")
            
            # Check for various issues
            file_info['issues'] = self.check_file_issues(content, relative_path)
            
            # Calculate complexity score
            if file_info['lines'] > 0:
                function_density = file_info['functions'] / max(1, file_info['lines'] / 10)
                class_density = file_info['classes'] / max(1, file_info['lines'] / 20)
                issue_penalty = len(file_info['issues']) * 0.1
                
                file_info['complexity_score'] = max(0.0, min(1.0, function_density + class_density - issue_penalty))
        
        except Exception as e:
            file_info['import_errors'].append(f"Could not read {relative_path}: {str(e)}")
        
        return file_info

    def check_file_issues(self, content, relative_path):
        """Check for common code issues"""
        issues = []
        
        for issue_type, patterns in self.issue_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    issues.append(f"{issue_type}: Line {line_num}")
        
        return issues

    def analyze_dependencies(self, build_path):
        """Analyze dependencies and imports"""
        dependencies = {
            'external_packages': set(),
            'internal_modules': set(),
            'missing_imports': [],
            'unused_imports': [],
            'dependency_graph': defaultdict(list)
        }
        
        python_files = list(build_path.rglob('*.py'))
        all_imports = set()
        internal_modules = set()
        
        # First pass: collect all imports and internal modules
        for py_file in python_files:
            module_name = py_file.stem
            internal_modules.add(module_name)
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find imports
                import_matches = re.finditer(r'^(?:from\s+(\S+)\s+import|import\s+(\S+))', content, re.MULTILINE)
                for match in import_matches:
                    module = match.group(1) or match.group(2)
                    if module:
                        module = module.split('.')[0]  # Get base module
                        all_imports.add(module)
                        dependencies['dependency_graph'][module_name].append(module)
            except:
                pass
        
        dependencies['internal_modules'] = internal_modules
        
        # Classify imports
        standard_lib = {'os', 'sys', 'json', 'datetime', 'time', 'random', 'math', 'collections', 're', 'pathlib', 'typing'}
        
        for import_name in all_imports:
            if import_name in standard_lib:
                continue
            elif import_name in internal_modules:
                continue
            else:
                dependencies['external_packages'].add(import_name)
        
        # Check for missing imports (imported but not available)
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                import_matches = re.finditer(r'^(?:from\s+(\S+)\s+import|import\s+(\S+))', content, re.MULTILINE)
                for match in import_matches:
                    module = match.group(1) or match.group(2)
                    if module and module.split('.')[0] in internal_modules:
                        # Check if the file exists
                        expected_file = build_path / f"{module.split('.')[0]}.py"
                        if not expected_file.exists():
                            dependencies['missing_imports'].append(f"{py_file.name} imports missing {module}")
            except:
                pass
        
        dependencies['external_packages'] = list(dependencies['external_packages'])
        dependencies['internal_modules'] = list(dependencies['internal_modules'])
        
        return dependencies

    def check_critical_files(self, build_path):
        """Check for presence of critical files"""
        critical_analysis = {
            'present': defaultdict(list),
            'missing': defaultdict(list),
            'completeness_score': 0.0,
            'entry_points': [],
            'config_files': []
        }
        
        all_files = [f.name.lower() for f in build_path.rglob('*') if f.is_file()]
        
        total_categories = 0
        present_categories = 0
        
        for category, file_list in self.critical_files.items():
            total_categories += 1
            category_present = False
            
            for critical_file in file_list:
                if critical_file.lower() in all_files:
                    critical_analysis['present'][category].append(critical_file)
                    category_present = True
                    
                    if category == 'main_entry':
                        critical_analysis['entry_points'].append(critical_file)
                    elif category == 'config':
                        critical_analysis['config_files'].append(critical_file)
                else:
                    critical_analysis['missing'][category].append(critical_file)
            
            if category_present:
                present_categories += 1
        
        critical_analysis['completeness_score'] = present_categories / total_categories if total_categories > 0 else 0.0
        
        return critical_analysis

    def identify_issues(self, build_path):
        """Identify potential issues and problems"""
        issues = {
            'critical': [],
            'warnings': [],
            'suggestions': [],
            'file_issues': [],
            'structure_issues': []
        }
        
        # Check for empty directories
        for folder in build_path.rglob('*'):
            if folder.is_dir() and not any(folder.iterdir()):
                issues['warnings'].append(f"Empty directory: {folder.relative_to(build_path)}")
        
        # Check for very large files (potential data dumps)
        for file_path in build_path.rglob('*'):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                if size_mb > 50:
                    issues['warnings'].append(f"Very large file: {file_path.name} ({size_mb:.1f} MB)")
                elif size_mb > 10:
                    issues['suggestions'].append(f"Large file: {file_path.name} ({size_mb:.1f} MB)")
        
        # Check for missing README or documentation
        has_readme = any(f.name.lower().startswith('readme') for f in build_path.iterdir() if f.is_file())
        if not has_readme:
            issues['suggestions'].append("No README file found")
        
        # Check for Python cache directories
        pycache_dirs = list(build_path.rglob('__pycache__'))
        if pycache_dirs:
            issues['suggestions'].append(f"Python cache directories present: {len(pycache_dirs)} directories")
        
        # Check for temporary files
        temp_files = []
        for file_path in build_path.rglob('*'):
            if file_path.is_file() and (file_path.name.startswith('.') or file_path.suffix in ['.tmp', '.bak', '.swp']):
                temp_files.append(file_path.name)
        
        if temp_files:
            issues['suggestions'].append(f"Temporary files found: {', '.join(temp_files[:5])}")
        
        return issues

    def analyze_sizes(self, build_path):
        """Analyze file and directory sizes"""
        size_analysis = {
            'total_size_mb': 0.0,
            'by_extension': {},
            'largest_files': [],
            'largest_directories': [],
            'size_distribution': {
                'tiny': 0,      # < 1KB
                'small': 0,     # 1KB - 10KB
                'medium': 0,    # 10KB - 100KB
                'large': 0,     # 100KB - 1MB
                'huge': 0       # > 1MB
            }
        }
        
        total_size = 0
        file_sizes = []
        
        # Analyze individual files
        for file_path in build_path.rglob('*'):
            if file_path.is_file() and file_path.name != 'BUILD_ANALYSIS_REPORT.md':
                size = file_path.stat().st_size
                total_size += size
                file_sizes.append((file_path, size))
                
                # Size distribution
                if size < 1024:
                    size_analysis['size_distribution']['tiny'] += 1
                elif size < 10240:
                    size_analysis['size_distribution']['small'] += 1
                elif size < 102400:
                    size_analysis['size_distribution']['medium'] += 1
                elif size < 1048576:
                    size_analysis['size_distribution']['large'] += 1
                else:
                    size_analysis['size_distribution']['huge'] += 1
        
        size_analysis['total_size_mb'] = total_size / (1024 * 1024)
        
        # Largest files
        file_sizes.sort(key=lambda x: x[1], reverse=True)
        size_analysis['largest_files'] = [
            {
                'name': f[0].name,
                'path': str(f[0].relative_to(build_path)),
                'size_mb': round(f[1] / (1024 * 1024), 3)
            }
            for f in file_sizes[:10]
        ]
        
        # Analyze directory sizes
        dir_sizes = []
        for dir_path in build_path.rglob('*'):
            if dir_path.is_dir():
                dir_size = sum(f.stat().st_size for f in dir_path.rglob('*') if f.is_file())
                dir_sizes.append((dir_path, dir_size))
        
        dir_sizes.sort(key=lambda x: x[1], reverse=True)
        size_analysis['largest_directories'] = [
            {
                'name': d[0].name,
                'path': str(d[0].relative_to(build_path)),
                'size_mb': round(d[1] / (1024 * 1024), 3)
            }
            for d in dir_sizes[:5]
        ]
        
        return size_analysis

    def calculate_completeness(self, analysis):
        """Calculate overall completeness score"""
        weights = {
            'critical_files': 0.3,
            'python_quality': 0.25,
            'organization': 0.2,
            'dependencies': 0.15,
            'issues': 0.1
        }
        
        # Critical files score
        critical_score = analysis['critical_files']['completeness_score']
        
        # Python quality score
        python_score = analysis['python_analysis']['code_quality_score']
        
        # Organization score
        org_score = analysis['folder_structure']['organization_score']
        
        # Dependencies score (fewer missing = better)
        missing_deps = len(analysis['dependencies']['missing_imports'])
        dep_score = max(0.0, 1.0 - (missing_deps * 0.1))
        
        # Issues score (fewer issues = better)
        total_issues = len(analysis['potential_issues']['critical']) + len(analysis['potential_issues']['warnings'])
        issue_score = max(0.0, 1.0 - (total_issues * 0.05))
        
        completeness = (
            critical_score * weights['critical_files'] +
            python_score * weights['python_quality'] +
            org_score * weights['organization'] +
            dep_score * weights['dependencies'] +
            issue_score * weights['issues']
        )
        
        return min(1.0, completeness)

    def generate_build_report(self, build_path, analysis):
        """Generate comprehensive markdown report"""
        report_path = build_path / "BUILD_ANALYSIS_REPORT.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self.format_report(analysis))

    def format_report(self, analysis):
        """Format the analysis data into a comprehensive markdown report"""
        build_info = analysis['build_info']
        file_analysis = analysis['file_analysis']
        folder_structure = analysis['folder_structure']
        python_analysis = analysis['python_analysis']
        dependencies = analysis['dependencies']
        critical_files = analysis['critical_files']
        issues = analysis['potential_issues']
        size_analysis = analysis['size_analysis']
        
        report = f"""# Build Analysis Report: {build_info['name']}

**Generated:** {build_info['analysis_date']}  
**Version:** {build_info['version']}  
**Tag:** {build_info['tag']}  
**Overall Completeness:** {analysis['completeness_score']:.1%}

---

## 📊 Executive Summary

| Metric | Value | Status |
|--------|-------|--------|
| Total Files | {file_analysis['total_count']} | {'✅' if file_analysis['total_count'] > 5 else '⚠️'} |
| Total Size | {size_analysis['total_size_mb']:.2f} MB | {'✅' if size_analysis['total_size_mb'] < 100 else '⚠️'} |
| Python Files | {len(python_analysis['files'])} | {'✅' if len(python_analysis['files']) > 0 else '❌'} |
| Critical Files | {len(critical_files['present'])} categories | {'✅' if critical_files['completeness_score'] > 0.6 else '⚠️'} |
| Code Quality | {python_analysis['code_quality_score']:.1%} | {'✅' if python_analysis['code_quality_score'] > 0.7 else '⚠️'} |
| Syntax Errors | {len(python_analysis['syntax_errors'])} | {'✅' if len(python_analysis['syntax_errors']) == 0 else '❌'} |

---

## 📁 File Type Analysis

| Extension | Count | Total Size | Avg Size | Largest File |
|-----------|-------|------------|----------|--------------|
"""
        
        # File type table
        for ext, data in sorted(file_analysis['by_extension'].items()):
            total_size_mb = data['total_size'] / (1024 * 1024)
            avg_size_kb = (data['total_size'] / data['count']) / 1024 if data['count'] > 0 else 0
            largest = max(data['files'], key=lambda x: x['size']) if data['files'] else {'name': 'N/A'}
            
            report += f"| {ext} | {data['count']} | {total_size_mb:.2f} MB | {avg_size_kb:.1f} KB | {largest['name']} |\n"
        
        report += f"""

---

## 🗂️ Folder Structure

**Organization Score:** {folder_structure['organization_score']:.1%}  
**Maximum Depth:** {folder_structure['depth']} levels  
**Total Folders:** {len(folder_structure['folders'])}

"""
        
        if folder_structure['folders']:
            report += "| Folder | Files | Subfolders | Path |\n|--------|-------|------------|------|\n"
            for folder in folder_structure['folders'][:10]:  # Show top 10
                report += f"| {folder['name']} | {folder['file_count']} | {folder['subfolder_count']} | {folder['path']} |\n"
        else:
            report += "- **Flat structure** (no subfolders)\n"
        
        report += f"""

---

## 🐍 Python Code Analysis

**Total Lines of Code:** {python_analysis['total_lines']:,}  
**Functions:** {python_analysis['total_functions']}  
**Classes:** {python_analysis['total_classes']}  
**Code Quality Score:** {python_analysis['code_quality_score']:.1%}

### Python Files Breakdown
| File | Size | Lines | Functions | Classes | Quality |
|------|------|-------|-----------|---------|---------|
"""
        
        for py_file in sorted(python_analysis['files'], key=lambda x: x['lines'], reverse=True)[:15]:
            size_kb = py_file['size'] / 1024
            quality_icon = '✅' if py_file['complexity_score'] > 0.7 else '⚠️' if py_file['complexity_score'] > 0.4 else '❌'
            report += f"| {py_file['name']} | {size_kb:.1f} KB | {py_file['lines']} | {py_file['functions']} | {py_file['classes']} | {quality_icon} |\n"
        
        # Syntax errors
        if python_analysis['syntax_errors']:
            report += f"\n### ❌ Syntax Errors ({len(python_analysis['syntax_errors'])})\n"
            for error in python_analysis['syntax_errors'][:10]:
                report += f"- {error}\n"
        
        report += f"""

---

## 🔗 Dependencies Analysis

**External Packages:** {len(dependencies['external_packages'])}  
**Internal Modules:** {len(dependencies['internal_modules'])}  
**Missing Imports:** {len(dependencies['missing_imports'])}

### External Dependencies
"""
        
        if dependencies['external_packages']:
            report += "```\n" + '\n'.join(sorted(dependencies['external_packages'])) + "\n```\n"
        else:
            report += "- No external dependencies detected\n"
        
        if dependencies['missing_imports']:
            report += f"\n### ⚠️ Missing Dependencies\n"
            for missing in dependencies['missing_imports'][:10]:
                report += f"- {missing}\n"
        
        report += f"""

---

## 🎯 Critical Files Assessment

**Completeness Score:** {critical_files['completeness_score']:.1%}

### Entry Points
"""
        
        if critical_files['entry_points']:
            for entry in critical_files['entry_points']:
                report += f"- ✅ {entry}\n"
        else:
            report += "- ❌ No clear entry point found\n"
        
        report += "\n### Core Components Status\n"
        for category, files in critical_files['present'].items():
            if files:
                report += f"- ✅ **{category.replace('_', ' ').title()}**: {', '.join(files)}\n"
        
        for category, files in critical_files['missing'].items():
            if files and not critical_files['present'][category]:
                report += f"- ❌ **{category.replace('_', ' ').title()}**: Missing all files\n"
        
        report += f"""

---

## ⚠️ Issues & Recommendations

### Critical Issues
"""
        
        if issues['critical']:
            for issue in issues['critical']:
                report += f"- 🚨 {issue}\n"
        else:
            report += "- No critical issues detected\n"
        
        report += "\n### Warnings\n"
        if issues['warnings']:
            for warning in issues['warnings'][:10]:
                report += f"- ⚠️ {warning}\n"
        else:
            report += "- No warnings\n"
        
        report += "\n### Suggestions\n"
        if issues['suggestions']:
            for suggestion in issues['suggestions'][:10]:
                report += f"- 💡 {suggestion}\n"
        else:
            report += "- No suggestions\n"
        
        report += f"""

---

## 📏 Size Analysis

**Total Build Size:** {size_analysis['total_size_mb']:.2f} MB

### Size Distribution
- **Tiny** (< 1KB): {size_analysis['size_distribution']['tiny']} files
- **Small** (1-10KB): {size_analysis['size_distribution']['small']} files  
- **Medium** (10-100KB): {size_analysis['size_distribution']['medium']} files
- **Large** (100KB-1MB): {size_analysis['size_distribution']['large']} files
- **Huge** (> 1MB): {size_analysis['size_distribution']['huge']} files

### Largest Files
| File | Size | Path |
|------|------|------|
"""
        
        for large_file in size_analysis['largest_files'][:10]:
            report += f"| {large_file['name']} | {large_file['size_mb']:.2f} MB | {large_file['path']} |\n"
        
        # Cleanup recommendations
        report += f"""

---

## 🧹 Cleanup Recommendations

### Priority Actions
"""
        
        cleanup_score = analysis['completeness_score']
        if cleanup_score > 0.8:
            report += "- ✅ **HIGH PRIORITY**: This build is well-organized and ready for use\n"
            report += "- Consider this build for production compilation\n"
        elif cleanup_score > 0.6:
            report += "- ⚠️ **MEDIUM PRIORITY**: Address missing dependencies and syntax errors\n"
            report += "- Good candidate for fixes and improvements\n"
        elif cleanup_score > 0.4:
            report += "- 🔧 **LOW PRIORITY**: Significant issues need resolution\n"
            report += "- Consider for feature extraction rather than full restoration\n"
        else:
            report += "- 🗑️ **CLEANUP CANDIDATE**: Multiple critical issues detected\n"
            report += "- Consider archiving or using for reference only\n"
        
        if file_analysis['empty_files']:
            report += f"- Remove {len(file_analysis['empty_files'])} empty files\n"
        
        if file_analysis['duplicate_files']:
            report += f"- Review {len(file_analysis['duplicate_files'])} sets of duplicate files\n"
        
        if any('__pycache__' in issue for issue in issues['suggestions']):
            report += "- Clean Python cache directories\n"
        
        if python_analysis['syntax_errors']:
            report += f"- Fix {len(python_analysis['syntax_errors'])} syntax errors\n"
        
        report += f"""

---

## 📋 Action Items

### Immediate Tasks
1. **Fix Syntax Errors**: {len(python_analysis['syntax_errors'])} files need attention
2. **Resolve Dependencies**: {len(dependencies['missing_imports'])} missing imports
3. **Code Review**: {len([f for f in python_analysis['files'] if f['complexity_score'] < 0.5])} files need improvement

### Long-term Tasks  
1. **Documentation**: Add README and code documentation
2. **Testing**: Create test suite for core functionality
3. **Optimization**: Review large files and optimize if necessary

### Verdict
**{build_info['name']}** - """
        
        if cleanup_score > 0.8:
            report += "**PRODUCTION READY** 🚀"
        elif cleanup_score > 0.6:
            report += "**GOOD CANDIDATE** ✅"
        elif cleanup_score > 0.4:
            report += "**NEEDS WORK** 🔧"
        else:
            report += "**CLEANUP NEEDED** 🗑️"
        
        report += f"""

---
*Report generated by V7P3R Build Analyzer on {build_info['analysis_date']}*
"""
        
        return report

if __name__ == "__main__":
    analyzer = IndividualBuildAnalyzer()
    analyzer.analyze_all_builds()
