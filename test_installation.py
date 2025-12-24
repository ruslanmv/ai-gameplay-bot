#!/usr/bin/env python3
"""
AI Gameplay Bot - Installation and Health Check Script
Tests project installation, dependencies, directory structure, and health.
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path
from typing import List, Tuple, Dict
import json

# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text:^70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓{Colors.RESET} {text}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗{Colors.RESET} {text}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {text}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ{Colors.RESET} {text}")

class HealthCheck:
    """Main health check class"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.results = {
            'passed': [],
            'failed': [],
            'warnings': []
        }

    def test_python_version(self) -> bool:
        """Test Python version compatibility"""
        print_info("Checking Python version...")
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            print_success(f"Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
            return True
        else:
            print_error(f"Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)")
            return False

    def test_required_packages(self) -> Tuple[bool, List[str]]:
        """Test if required packages are installed"""
        print_info("Checking required Python packages...")

        required_packages = [
            'torch',
            'torchvision',
            'transformers',
            'numpy',
            'opencv-python',
            'Pillow',
            'pandas',
            'scikit-learn',
            'scipy',
            'Flask',
            'Flask-CORS',
            'requests',
            'gunicorn',
            'gym',
            'stable-baselines3',
            'python-dotenv',
            'tqdm',
            'pyyaml',
            'colorama',
            'prometheus-client',
            'psutil',
            'pytest',
            'pytest-cov',
            'black',
            'flake8',
            'pylint'
        ]

        missing = []
        installed = []

        for package in required_packages:
            # Convert package name to import name
            import_name = package.replace('-', '_').lower()

            # Special cases
            if package == 'opencv-python':
                import_name = 'cv2'
            elif package == 'Pillow':
                import_name = 'PIL'
            elif package == 'python-dotenv':
                import_name = 'dotenv'
            elif package == 'pyyaml':
                import_name = 'yaml'
            elif package == 'scikit-learn':
                import_name = 'sklearn'

            try:
                importlib.import_module(import_name)
                installed.append(package)
                print_success(f"{package:30s} installed")
            except ImportError:
                missing.append(package)
                print_error(f"{package:30s} missing")

        if missing:
            print_error(f"\nMissing {len(missing)} packages: {', '.join(missing)}")
            print_info("Run 'make install' or 'pip install -r requirements.txt' to install")
            return False, missing
        else:
            print_success(f"\nAll {len(installed)} required packages installed")
            return True, []

    def test_directory_structure(self) -> bool:
        """Test if required directories exist"""
        print_info("Checking directory structure...")

        required_dirs = [
            'models',
            'models/neural_network',
            'models/transformer',
            'deployment',
            'scripts',
            'frontend',
            'tests',
            'data',
            'data/raw',
            'data/processed'
        ]

        all_exist = True
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists():
                print_success(f"Directory exists: {dir_path}")
            else:
                print_warning(f"Directory missing: {dir_path}")
                all_exist = False

        return all_exist

    def test_frontend_files(self) -> bool:
        """Test if frontend files exist"""
        print_info("Checking frontend files...")

        frontend_files = [
            'frontend/index.html'
        ]

        all_exist = True
        for file_path in frontend_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                size = full_path.stat().st_size
                print_success(f"Frontend file exists: {file_path} ({size:,} bytes)")
            else:
                print_error(f"Frontend file missing: {file_path}")
                all_exist = False

        return all_exist

    def test_deployment_files(self) -> bool:
        """Test if deployment files exist"""
        print_info("Checking deployment files...")

        deployment_files = [
            'deployment/control_backend.py',
            'deployment/deploy_nn.py',
            'deployment/deploy_transformer.py'
        ]

        all_exist = True
        for file_path in deployment_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                print_success(f"Deployment file exists: {file_path}")
            else:
                print_error(f"Deployment file missing: {file_path}")
                all_exist = False

        return all_exist

    def test_model_files(self) -> bool:
        """Test if model training files exist"""
        print_info("Checking model training files...")

        model_files = [
            'models/neural_network/nn_training.py',
            'models/transformer/transformer_training.py'
        ]

        all_exist = True
        for file_path in model_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                print_success(f"Model file exists: {file_path}")
            else:
                print_error(f"Model file missing: {file_path}")
                all_exist = False

        return all_exist

    def test_pytest(self) -> bool:
        """Run pytest if available"""
        print_info("Running pytest tests...")

        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short'],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                print_success("All pytest tests passed")
                # Print test output
                for line in result.stdout.split('\n'):
                    if 'PASSED' in line:
                        print(f"  {Colors.GREEN}✓{Colors.RESET} {line.strip()}")
                    elif 'FAILED' in line:
                        print(f"  {Colors.RED}✗{Colors.RESET} {line.strip()}")
                return True
            else:
                print_error("Some pytest tests failed")
                print("\nTest output:")
                print(result.stdout)
                if result.stderr:
                    print("\nErrors:")
                    print(result.stderr)
                return False
        except FileNotFoundError:
            print_warning("pytest not found - skipping tests")
            return True
        except subprocess.TimeoutExpired:
            print_error("Tests timed out after 60 seconds")
            return False
        except Exception as e:
            print_error(f"Error running tests: {e}")
            return False

    def test_makefile_targets(self) -> bool:
        """Test if Makefile exists and has required targets"""
        print_info("Checking Makefile...")

        makefile = self.project_root / 'Makefile'
        if not makefile.exists():
            print_error("Makefile not found")
            return False

        required_targets = ['install', 'test', 'setup', 'run-control', 'clean']
        content = makefile.read_text()

        all_exist = True
        for target in required_targets:
            if f"{target}:" in content:
                print_success(f"Makefile target exists: {target}")
            else:
                print_error(f"Makefile target missing: {target}")
                all_exist = False

        return all_exist

    def test_requirements_file(self) -> bool:
        """Test if requirements.txt exists"""
        print_info("Checking requirements.txt...")

        req_file = self.project_root / 'requirements.txt'
        if req_file.exists():
            lines = req_file.read_text().strip().split('\n')
            # Count non-empty, non-comment lines
            packages = [l for l in lines if l.strip() and not l.strip().startswith('#')]
            print_success(f"requirements.txt exists with {len(packages)} packages")
            return True
        else:
            print_error("requirements.txt not found")
            return False

    def generate_report(self):
        """Generate final health check report"""
        print_header("HEALTH CHECK SUMMARY")

        total_tests = len(self.results['passed']) + len(self.results['failed'])
        passed = len(self.results['passed'])
        failed = len(self.results['failed'])
        warnings = len(self.results['warnings'])

        print(f"\n{Colors.BOLD}Total Tests:{Colors.RESET}     {total_tests}")
        print(f"{Colors.GREEN}{Colors.BOLD}Passed:{Colors.RESET}         {passed}")
        print(f"{Colors.RED}{Colors.BOLD}Failed:{Colors.RESET}         {failed}")
        print(f"{Colors.YELLOW}{Colors.BOLD}Warnings:{Colors.RESET}       {warnings}\n")

        if failed > 0:
            print(f"\n{Colors.RED}{Colors.BOLD}Failed Tests:{Colors.RESET}")
            for test in self.results['failed']:
                print(f"  {Colors.RED}✗{Colors.RESET} {test}")

        if warnings > 0:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}Warnings:{Colors.RESET}")
            for warning in self.results['warnings']:
                print(f"  {Colors.YELLOW}⚠{Colors.RESET} {warning}")

        # Success rate
        if total_tests > 0:
            success_rate = (passed / total_tests) * 100
            print(f"\n{Colors.BOLD}Success Rate:{Colors.RESET} {success_rate:.1f}%\n")

            if success_rate == 100:
                print(f"{Colors.GREEN}{Colors.BOLD}✓ All tests passed! Project is healthy.{Colors.RESET}\n")
                return 0
            elif success_rate >= 80:
                print(f"{Colors.YELLOW}{Colors.BOLD}⚠ Most tests passed. Review warnings.{Colors.RESET}\n")
                return 1
            else:
                print(f"{Colors.RED}{Colors.BOLD}✗ Multiple tests failed. Fix issues above.{Colors.RESET}\n")
                return 2
        return 0

    def run_all_tests(self):
        """Run all health checks"""
        print_header("AI GAMEPLAY BOT - INSTALLATION & HEALTH CHECK")

        # Test 1: Python version
        print_header("1. Python Version")
        if self.test_python_version():
            self.results['passed'].append('Python version')
        else:
            self.results['failed'].append('Python version')

        # Test 2: Required packages
        print_header("2. Python Packages")
        success, missing = self.test_required_packages()
        if success:
            self.results['passed'].append('Python packages')
        else:
            self.results['failed'].append(f'Python packages ({len(missing)} missing)')

        # Test 3: Requirements file
        print_header("3. Requirements File")
        if self.test_requirements_file():
            self.results['passed'].append('requirements.txt')
        else:
            self.results['failed'].append('requirements.txt')

        # Test 4: Makefile
        print_header("4. Makefile")
        if self.test_makefile_targets():
            self.results['passed'].append('Makefile')
        else:
            self.results['failed'].append('Makefile')

        # Test 5: Directory structure
        print_header("5. Directory Structure")
        if self.test_directory_structure():
            self.results['passed'].append('Directory structure')
        else:
            self.results['warnings'].append('Some directories missing (will be created on setup)')

        # Test 6: Frontend files
        print_header("6. Frontend Files")
        if self.test_frontend_files():
            self.results['passed'].append('Frontend files')
        else:
            self.results['failed'].append('Frontend files')

        # Test 7: Deployment files
        print_header("7. Deployment Files")
        if self.test_deployment_files():
            self.results['passed'].append('Deployment files')
        else:
            self.results['failed'].append('Deployment files')

        # Test 8: Model files
        print_header("8. Model Training Files")
        if self.test_model_files():
            self.results['passed'].append('Model files')
        else:
            self.results['failed'].append('Model files')

        # Test 9: Pytest tests
        print_header("9. Running Tests")
        if self.test_pytest():
            self.results['passed'].append('Pytest tests')
        else:
            self.results['failed'].append('Pytest tests')

        # Generate final report
        return self.generate_report()

def main():
    """Main entry point"""
    health_check = HealthCheck()
    exit_code = health_check.run_all_tests()

    print_info("\nNext steps:")
    if exit_code == 0:
        print("  • Run 'make run-control' to start the backend")
        print("  • Open frontend/index.html to access the dashboard")
    elif exit_code == 1:
        print("  • Review warnings above")
        print("  • Run 'make setup' to create missing directories")
    else:
        print("  • Run 'make install' to install dependencies")
        print("  • Run 'make setup' to create project structure")
        print("  • Fix errors listed above")

    sys.exit(exit_code)

if __name__ == '__main__':
    main()
