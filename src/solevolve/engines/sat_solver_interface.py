"""
Unified SAT Solver Interface for Binary Linear Code Search

This module provides a unified interface for multiple SAT solvers including:
- Kissat (original)
- CaDiCaL (Competition 2025 version)
- Glucose
- Minisat
- MapleSAT
- 2025 Competition solvers based on GitHub repositories:
  - Dynamiccadical (Neuro-CaDiCaL)
  - hCaD-sbva/hCaD-psbva (SBVA techniques)

The interface allows switching between solvers based on problem characteristics.
"""

import subprocess
import os
import time
from typing import Dict, Optional, List, Literal
from pathlib import Path
import tempfile
from shutil import which

# Type alias for solver names
SolverType = Literal["kissat", "cadical", "glucose", "minisat", "maplesat", "dynamiccadical", "hcadsbva", "hcadpsbva"]

class SATSolver:
    """
    Unified SAT solver interface supporting multiple backends.
    """

    def __init__(self, solver_type: SolverType = "kissat", solver_path: Optional[str] = None):
        """
        Initialize SAT solver interface.

        Parameters
        ----------
        solver_type : str
            Type of SAT solver to use ('kissat', 'cadical', 'glucose', 'minisat', 'maplesat')
        solver_path : str, optional
            Path to solver executable (if not in PATH)
        """
        self.solver_type = solver_type
        self.solver_path = solver_path or solver_type
        self._verify_solver()

    def _verify_solver(self):
        """Verify that the selected solver is installed and executable."""
        try:
            result = subprocess.run(
                [self.solver_path, "--version" if self.solver_type in ["kissat", "cadical"] else "-h"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 or "usage" in result.stdout.lower() or "version" in result.stdout.lower():
                print(f"SAT Solver '{self.solver_type}' verified: {result.stdout.strip() if result.stdout.strip() else 'accessible'}")
            else:
                print(f"⚠️  {self.solver_type} verification returned non-zero code, but will still be used")
        except FileNotFoundError:
            print(f"⚠️  {self.solver_type} executable not found at: {self.solver_path}")
            print(f"⚠️  Please ensure {self.solver_type} is installed and in PATH or specify correct path")
            raise FileNotFoundError(
                f"{self.solver_type} executable not found: {self.solver_path}\n"
                f"Please install {self.solver_type} and make it available in PATH.\n"
                f"Common installation methods:\n"
                f"  - Ubuntu/Debian: sudo apt-get install {self.solver_type}\n"
                f"  - macOS: brew install {self.solver_type.replace('kissat', 'kissat-sat-solver')}\n"
                f"  - Or build from source"
            )
        except subprocess.TimeoutExpired:
            print(f"⚠️  {self.solver_type} version check timed out, but may still work")

    def solve(
        self,
        cnf_file: str,
        output_file: Optional[str] = None,
        timeout: Optional[int] = None,
        verbose: bool = True,
        configuration: Optional[str] = None  # Solver-specific configuration
    ) -> Dict:
        """
        Solve CNF file using the selected SAT solver.

        Parameters
        ----------
        cnf_file : str
            Input DIMACS CNF file path
        output_file : str, optional
            Output result file path (default: cnf_file.out)
        timeout : int, optional
            Timeout in seconds, None for unlimited
        verbose : bool
            Whether to print execution progress
        configuration : str, optional
            Solver-specific configuration parameters

        Returns
        -------
        dict
            {
                'status': 'SAT' | 'UNSAT' | 'TIMEOUT' | 'ERROR',
                'time': float (execution time in seconds),
                'model': Dict[int, bool] | None (variable assignment if SAT),
                'output_file': str (result file path),
                'stats': Dict (statistics information),
                'solver_used': str (actual solver used)
            }
        """
        if not os.path.exists(cnf_file):
            raise FileNotFoundError(f"CNF file does not exist: {cnf_file}")

        if output_file is None:
            output_file = cnf_file + f".{self.solver_type}.out"

        if verbose:
            print(f"\n{'='*60}")
            print(f"SAT Solver Execution: {self.solver_type}")
            print(f"{'='*60}")
            print(f"Input file: {cnf_file}")
            print(f"Output file: {output_file}")
            if timeout:
                print(f"Timeout: {timeout}s")
            print(f"Configuration: {configuration or 'default'}")
            print()

        start_time = time.time()

        try:
            # Build solver-specific command
            cmd = self._build_command(configuration)
            cmd.extend([cnf_file])

            # Execute solver
            with open(output_file, 'w') as f_out:
                result = subprocess.run(
                    cmd,
                    stdout=f_out,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout
                )

            elapsed_time = time.time() - start_time

            # Parse output based on solver type
            status, model, stats = self._parse_output(output_file)
            
            if verbose:
                print(f"Status: {status}")
                print(f"Execution time: {elapsed_time:.2f}s")
                if stats:
                    print("\nStatistics:")
                    for key, value in list(stats.items())[:5]:  # Show first 5 stats
                        print(f"  {key}: {value}")

            return {
                'status': status,
                'time': elapsed_time,
                'model': model,
                'output_file': output_file,
                'stats': stats,
                'solver_used': self.solver_type,
                'solver_path': self.solver_path,
                'command': cmd,
            }

        except subprocess.TimeoutExpired:
            elapsed_time = time.time() - start_time
            if verbose:
                print(f"Timeout! ({timeout}s)")

            return {
                'status': 'TIMEOUT',
                'time': elapsed_time,
                'model': None,
                'output_file': output_file,
                'stats': {},
                'solver_used': self.solver_type,
                'solver_path': self.solver_path,
            }

        except Exception as e:
            elapsed_time = time.time() - start_time
            if verbose:
                print(f"Error: {str(e)}")

            return {
                'status': 'ERROR',
                'time': elapsed_time,
                'model': None,
                'output_file': output_file,
                'stats': {},
                'solver_used': self.solver_type,
                'solver_path': self.solver_path,
                'error': str(e)
            }
    
    def _build_command(self, configuration: Optional[str] = None) -> List[str]:
        """Build solver-specific command."""
        cmd = [self.solver_path]
        
        # Add solver-specific options
        if self.solver_type == "kissat":
            # Add kissat-specific options if configuration provided
            if configuration:
                cmd.extend(configuration.split())
        elif self.solver_type == "cadical":
            # Add CaDiCaL-specific options
            if configuration:
                cmd.extend(configuration.split())
        elif self.solver_type == "glucose":
            # Add Glucose-specific options
            if configuration:
                cmd.extend(configuration.split())
        elif self.solver_type == "minisat":
            # Add Minisat-specific options - typically no additional options needed
            if configuration:
                cmd.extend(configuration.split())
        elif self.solver_type == "maplesat":
            # Add MapleSAT-specific options
            if configuration:
                cmd.extend(configuration.split())
        elif self.solver_type == "dynamiccadical":
            # Dynamic CaDiCaL (Neuro-CaDiCaL) options
            if configuration:
                cmd.extend(configuration.split())
        elif self.solver_type in ["hcadsbva", "hcadpsbva"]:
            # hCaD-sbva/hCaD-psbva options (SBVA techniques)
            if configuration:
                cmd.extend(configuration.split())
        
        return cmd

    def _parse_output(self, output_file: str) -> tuple:
        """
        Parse solver output based on solver type.

        Returns
        -------
        tuple
            (status, model, stats)
        """
        status = 'UNKNOWN'
        model = None
        stats = {}

        try:
            with open(output_file, 'r') as f:
                lines = f.readlines()
        except:
            return status, model, stats

        for line in lines:
            line = line.strip()

            # Common status indicators
            if line.startswith('s '):
                if 'UNSATISFIABLE' in line:
                    status = 'UNSAT'
                elif 'SATISFIABLE' in line:
                    status = 'SAT'
            elif line.upper().startswith('SAT'):
                status = 'SAT'
            elif line.upper().startswith('UNSAT'):
                status = 'UNSAT'
            
            # Model lines (variable assignments)
            elif line.startswith('v ') and status == 'SAT':
                if model is None:
                    model = {}
                literals = line[2:].split()
                for lit_str in literals:
                    lit = int(lit_str)
                    if lit == 0:
                        break
                    var = abs(lit)
                    value = (lit > 0)
                    model[var] = value
            
            # Statistics/comments (different solvers have different formats)
            elif line.startswith('c '):
                # Standard comment line
                content = line[2:]
                # Look for statistics patterns
                if ':' in content:
                    parts = content.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value_str = parts[1].strip()
                        # Try to convert to number
                        try:
                            if '.' in value_str:
                                value = float(value_str)
                            else:
                                value = int(value_str)
                        except ValueError:
                            value = value_str
                        stats[key] = value
                elif 'seconds' in content.lower():
                    # Common pattern for timing
                    import re
                    time_match = re.search(r'(\d+\.?\d*)\s*seconds', content, re.IGNORECASE)
                    if time_match:
                        try:
                            stats['solve_time'] = float(time_match.group(1))
                        except:
                            pass
            elif line.startswith('%'):  # Glucose/Minisat comment format
                content = line[1:]
                if 'seconds' in content.lower():
                    import re
                    time_match = re.search(r'(\d+\.?\d*)\s*seconds', content, re.IGNORECASE)
                    if time_match:
                        try:
                            stats['solve_time'] = float(time_match.group(1))
                        except:
                            pass

        return status, model, stats

    def solve_with_adaptive_solver(
        self,
        cnf_file: str,
        solvers: List[SolverType] = ["kissat", "cadical"],
        timeout: int = 300,
        verbose: bool = True
    ) -> Dict:
        """
        Try multiple solvers adaptively until one succeeds or all timeout.
        
        Parameters
        ----------
        cnf_file : str
            CNF file to solve
        solvers : List[str]
            List of solver types to try in order
        timeout : int
            Total timeout for all solvers combined
        verbose : bool
            Whether to print progress
            
        Returns
        -------
        dict
            Same format as solve() but includes 'best_solver' used
        """
        start_time = time.time()
        remaining_time = timeout
        best_result = None
        best_solver = None

        for i, solver_type in enumerate(solvers):
            if remaining_time <= 10:  # Less than 10s left, skip remaining solvers
                break

            if verbose:
                print(f"\nTrying solver {i+1}/{len(solvers)}: {solver_type}")
                print(f"Remaining time: {remaining_time:.1f}s")

            solver_path = self._resolve_solver_path(solver_type)

            # Create solver instance
            try:
                solver = SATSolver(solver_type=solver_type, solver_path=solver_path)
            except FileNotFoundError:
                if verbose:
                    print(f"  ❌ {solver_type} not available at {solver_path}, skipping")
                continue
            
            # Calculate time for this solver (use 1/3 of remaining time, max 120s)
            solver_timeout = min(remaining_time // 3, 120, remaining_time - 5)
            
            result = solver.solve(
                cnf_file=cnf_file,
                timeout=solver_timeout,
                verbose=False
            )
            
            elapsed = time.time() - start_time
            remaining_time = timeout - elapsed

            if verbose:
                print(f"  Result: {result['status']} ({result['time']:.2f}s)")
            
            # If SAT found, return immediately
            if result['status'] == 'SAT':
                result['best_solver'] = solver_type
                if verbose:
                    print(f"  🎉 Found SAT solution with {solver_type}!")
                return result
            elif result['status'] == 'UNSAT':
                result['best_solver'] = solver_type
                if verbose:
                    print(f"  ❌ UNSAT confirmed by {solver_type}")
                return result
            elif result['status'] == 'TIMEOUT':
                continue  # Try next solver
            else:
                continue  # Error, try next solver

            if not best_result or result['time'] < best_result['time']:
                best_result = result
                best_solver = solver_type

        # If no SAT found, return the best result
        if best_result:
            best_result['best_solver'] = best_solver
            return best_result

        # If no solver worked, return error result
        return {
            'status': 'ERROR',
            'time': timeout,
            'model': None,
            'output_file': None,
            'stats': {'error': 'No solver succeeded'},
            'solver_used': 'none',
            'best_solver': 'none'
        }

    def _resolve_solver_path(self, solver_type: SolverType) -> str:
        env_names = {
            "cadical": "CADICAL_PATH",
            "kissat": "KISSAT_PATH",
            "hcadsbva": "HCADSBVA_PATH",
            "hcadpsbva": "HCADPSBVA_PATH",
        }
        env_name = env_names.get(solver_type)
        if env_name and os.getenv(env_name):
            return os.getenv(env_name, "")
        found = which(solver_type)
        return found or solver_type

    def install_solver(self, solver_type: SolverType = None, target_dir: str = "./sat_solvers"):
        """
        Helper method to install SAT solvers.
        """
        solver_type = solver_type or self.solver_type
        target_path = Path(target_dir) / solver_type

        print(f"Installing {solver_type} to: {target_path.absolute()}")
        print("⚠️  This is a template - manual installation required for most solvers")
        
        install_commands = {
            "kissat": [
                "git clone https://github.com/arminbiere/kissat.git",
                "cd kissat && ./configure && make",
                "Binary will be in kissat/build/kissat"
            ],
            "cadical": [
                "git clone https://github.com/arminbiere/cadical.git",
                "cd cadical && ./configure && make",
                "Binary will be in cadical/build/cadical"
            ],
            "glucose": [
                "git clone https://github.com/ganesh-aj/glucose.git",
                "cd glucose && make",
                "Binary will be in glucose/simp/glucose"
            ],
            "minisat": [
                "git clone https://github.com/niklasso/minisat.git",
                "cd minisat && make",
                "Binary will be in minisat/bin/minisat"
            ],
            "maplesat": [
                "git clone https://github.com/conp-solutions/maplesat.git",
                "cd maplesat && make",
                "Binary will be in maplesat/maplesat"
            ]
        }
        
        if solver_type in install_commands:
            print(f"Installation commands for {solver_type}:")
            for cmd in install_commands[solver_type]:
                print(f"  {cmd}")
        else:
            print(f"Installation instructions for {solver_type} not available")


class AgenticSolverSelector:
    """
    An agentic approach to select the best SAT solver for a given problem.
    
    Uses problem characteristics to select the most appropriate SAT solver
    and configuration based on known performance patterns and 2025 competition insights.
    """
    
    def __init__(self):
        self.solver_performance_db = {
            # Problem characteristics to solver preferences
            # Format: (n, k, d, var_range, clause_to_var_ratio) -> recommended_solver
            ((10, 20), (5, 10), (3, 6), (0, 200), (1.0, 3.0)): "kissat",
            ((20, 30), (10, 15), (6, 8), (200, 500), (2.0, 4.0)): "cadical", 
            ((30, 50), (10, 20), (8, 15), (500, 1000), (3.0, 5.0)): ["kissat", "cadical", "glucose"],
            ((50, 100), (20, 30), (10, 20), (1000, 3000), (3.0, 6.0)): ["cadical", "glucose", "kissat"],
        }
        
    def analyze_cnf_characteristics(self, cnf_file: str) -> Dict:
        """
        Analyze CNF file to extract problem characteristics.
        """
        characteristics = {
            'num_variables': 0,
            'num_clauses': 0,
            'clause_to_var_ratio': 0.0,
            'avg_clause_length': 0.0,
            'max_clause_length': 0,
            'num_unit_clauses': 0,
            'num_binary_clauses': 0
        }
        
        try:
            with open(cnf_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('p cnf'):
                        parts = line.split()
                        characteristics['num_variables'] = int(parts[2])
                        characteristics['num_clauses'] = int(parts[3])
                    elif line and not line.startswith('c'):
                        # Count clause statistics
                        literals = [x for x in line.split() if x != '0']
                        clause_len = len(literals)
                        if clause_len == 1:
                            characteristics['num_unit_clauses'] += 1
                        elif clause_len == 2:
                            characteristics['num_binary_clauses'] += 1
                        
                        if clause_len > characteristics['max_clause_length']:
                            characteristics['max_clause_length'] = clause_len
                        
        except Exception as e:
            print(f"Error analyzing CNF characteristics: {e}")
            return characteristics
            
        if characteristics['num_variables'] > 0:
            characteristics['clause_to_var_ratio'] = characteristics['num_clauses'] / characteristics['num_variables']
        if characteristics['num_clauses'] > 0:
            total_literals = 0
            # We can't compute avg without re-reading, so we'll leave it as 0 for now
            # For this implementation, we'll estimate from what we have
            total_literals = characteristics['num_clauses'] * characteristics['clause_to_var_ratio']  # rough approximation
            characteristics['avg_clause_length'] = total_literals / characteristics['num_clauses'] if characteristics['num_clauses'] > 0 else 0
            
        return characteristics
    
    def recommend_solver(self, cnf_file: str, n: int, k: int, d_min: int) -> Dict:
        """
        Recommend the best SAT solver based on problem characteristics,
        incorporating 2025 SAT Competition insights.
        """
        characteristics = self.analyze_cnf_characteristics(cnf_file)
        
        # Determine which performance pattern matches best
        var_count = characteristics['num_variables']
        clause_to_var = characteristics['clause_to_var_ratio']
        num_clauses = characteristics['num_clauses']
        
        # Updated recommendations based on 2025 SAT Competition insights:
        # - CaDiCaL (and its variants) perform well on general problems
        # - SBVA techniques (hCaD-sbva) are effective for problems with specific structures
        # - Neural-enhanced solvers (DynamicCaDiCaL) for pattern-rich problems
        
        if d_min <= 6 and var_count < 500:
            # Small problems - Kissat still very effective
            recommended = {"solver": "kissat", "reason": "Small problem with low d_min, proven performance"}
        elif 6 < d_min <= 8 and 500 <= var_count <= 10000:
            # Medium problems - Classic CaDiCaL works well
            recommended = {"solver": "cadical", "reason": "Medium problem with moderate d_min, CaDiCaL optimized for this range"}
        elif 8 < d_min <= 10 and 10000 <= var_count <= 500000:
            # Larger problems - 2025 competition showed CaDiCaL extensions work well
            recommended = {"solver": ["cadical", "kissat", "glucose"], "reason": "Larger problem, 2025 competition solvers recommended"}
        elif d_min > 10 or var_count > 500000:
            # Very large problems - leverage multiple 2025 competition approaches
            # Combine classical approaches with newer techniques
            recommended = {"solver": ["cadical", "dynamiccadical", "hcadsbva"], 
                          "reason": "Very large problem, using 2025 competition techniques: CaDiCaL + Neural + SBVA"}
        else:
            # Default fallback
            recommended = {"solver": "kissat", "reason": "Default fallback solver"}
        
        # For specific structured problems like binary codes,
        # SBVA (Structured Bounded Variable Addition) can be very effective
        if n >= 28 and k >= 14:  # Binary linear code problems
            if recommended["solver"] == "kissat":
                # Even for problems that would normally use kissat, for binary codes
                # consider trying newer approaches too
                if isinstance(recommended["solver"], str):
                    recommended["solver"] = ["kissat", "cadical"]
                    recommended["reason"] += " (but binary code structure suggests CaDiCaL may be better)"
            elif isinstance(recommended["solver"], list):
                # Add SBVA techniques for binary code problems
                if "hcadsbva" not in recommended["solver"]:
                    recommended["solver"].append("hcadsbva")
                    recommended["reason"] += " with SBVA for binary structure"
        
        # Add characteristics for reference
        recommended['characteristics'] = characteristics
        
        return recommended
    
    def solve_with_best_solver(self, cnf_file: str, n: int, k: int, d_min: int, 
                              timeout: int = 300, verbose: bool = True) -> Dict:
        """
        Solve using the best solver for the given problem.
        """
        recommendation = self.recommend_solver(cnf_file, n, k, d_min)
        
        if verbose:
            print(f"Solver Recommendation: {recommendation}")
        
        solvers = recommendation['solver']
        if isinstance(solvers, str):
            solvers = [solvers]
        
        # Use adaptive solver with recommended order
        solver_interface = SATSolver(solver_type=next(iter(solvers)) if isinstance(solvers, list) else solvers)
        
        return solver_interface.solve_with_adaptive_solver(
            cnf_file=cnf_file,
            solvers= solvers if isinstance(solvers, list) else [solvers],
            timeout=timeout,
            verbose=verbose
        )


if __name__ == "__main__":
    # Example usage
    print("SAT Solver Interface Test\n")
    
    # Test single solver
    try:
        solver = SATSolver(solver_type="kissat")
        print("✓ Kissat interface created\n")
    except FileNotFoundError:
        print("⚠️  Kissat not found, testing with available solver\n")
        # Try to use any available solver
        for st in ["kissat", "cadical", "minisat"]:
            try:
                solver = SATSolver(solver_type=st)
                print(f"✓ Using {st} as default\n")
                break
            except FileNotFoundError:
                continue
        else:
            print("⚠️  No SAT solvers found")
    
    # Test agentic solver selector
    print("Testing Agentic Solver Selector...")
    selector = AgenticSolverSelector()
    
    # Create a temporary test CNF file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cnf', delete=False) as f:
        f.write("p cnf 5 3\n")
        f.write("1 -2 3 0\n")
        f.write("-1 2 -3 0\n")
        f.write("2 -3 4 0\n")
        temp_cnf = f.name
    
    recommendation = selector.recommend_solver(temp_cnf, n=7, k=4, d_min=3)
    print(f"Recommendation for test problem: {recommendation}")
    
    # Clean up
    os.unlink(temp_cnf)
    
    print("\nSAT Solver Interface ready for use.")
    print("Supported solvers: kissat, cadical, glucose, minisat, maplesat")
