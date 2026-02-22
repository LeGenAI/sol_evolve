"""
Code Generator - 통합 워크플로우

이 모듈은 CNF 인코딩, Kissat 실행, 결과 파싱을 하나의 워크플로우로 통합합니다.
"""

import os
import numpy as np
from typing import Dict, Optional
from pathlib import Path

from .cnf_encoder import CNFEncoder
from .sat_solver_interface import SATSolver, AgenticSolverSelector
from .result_parser import ResultParser


class CodeGenerator:
    """
    Binary Linear Code를 SAT 기반으로 탐색하고 생성하는 통합 클래스.
    """

    def __init__(
        self,
        n: int,
        k: int,
        d_min: int,
        systematic: bool = True,
        working_dir: str = "./workspace"
    ):
        """
        Parameters
        ----------
        n : int
            부호어 길이
        k : int
            메시지 길이
        d_min : int
            최소 거리
        systematic : bool
            체계적 형태 사용 여부
        working_dir : str
            작업 디렉토리 (CNF 파일, 결과 파일 저장)
        """
        self.n = n
        self.k = k
        self.d_min = d_min
        self.systematic = systematic
        self.working_dir = Path(working_dir)

        # 작업 디렉토리 생성
        self.working_dir.mkdir(parents=True, exist_ok=True)

        # 코드 이름
        self.code_name = f"code_{n}_{k}_{d_min}"

        # 컴포넌트 초기화
        self.encoder = CNFEncoder(n, k, d_min, systematic)
        self.parser = ResultParser(n, k, d_min, systematic)
        self.solver = None  # Kissat 경로가 제공될 때 초기화

        print(f"\n{'='*60}")
        print(f"Code Generator 초기화")
        print(f"{'='*60}")
        print(f"목표: [{n}, {k}, {d_min}] Binary Linear Code")
        print(f"체계적 형태: {systematic}")
        print(f"작업 디렉토리: {self.working_dir.absolute()}")
        print(f"{'='*60}\n")

    def generate_cnf(self, output_file: Optional[str] = None) -> str:
        """
        제약 조건을 CNF로 인코딩하고 파일로 저장.

        Parameters
        ----------
        output_file : str, optional
            CNF 파일 경로 (기본값: working_dir/code_name.cnf)

        Returns
        -------
        str
            생성된 CNF 파일 경로
        """
        if output_file is None:
            output_file = str(self.working_dir / f"{self.code_name}.cnf")

        print(f"CNF 생성 중...")
        self.encoder.encode_all_constraints()
        self.encoder.export_to_dimacs(output_file)

        return output_file

    def solve(
        self,
        kissat_path: str = "kissat",
        cnf_file: Optional[str] = None,
        timeout: Optional[int] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Kissat으로 SAT 문제를 풀고 결과를 반환.

        Parameters
        ----------
        kissat_path : str
            Kissat 실행 파일 경로
        cnf_file : str, optional
            입력 CNF 파일 (없으면 자동 생성)
        timeout : int, optional
            제한 시간(초)
        verbose : bool
            상세 출력 여부

        Returns
        -------
        dict
            {
                'status': 'SAT' | 'UNSAT' | 'TIMEOUT' | 'ERROR',
                'generator_matrix': np.ndarray | None,
                'verification': dict | None,
                'cnf_file': str,
                'solver_time': float,
                'total_time': float
            }
        """
        import time
        start_time = time.time()

        # CNF 파일 생성 (없으면)
        if cnf_file is None:
            cnf_file = self.generate_cnf()
        elif not os.path.exists(cnf_file):
            print(f"CNF 파일이 존재하지 않습니다. 생성합니다...")
            cnf_file = self.generate_cnf(cnf_file)

        # SAT 솔버 초기화 - using agentic selection
        self.solver = SATSolver(solver_type="kissat", solver_path=kissat_path)

        # SAT 문제 풀기
        sat_result = self.solver.solve(cnf_file, timeout=timeout, verbose=verbose)

        result = {
            'status': sat_result['status'],
            'generator_matrix': None,
            'verification': None,
            'cnf_file': cnf_file,
            'solver_time': sat_result['time'],
            'total_time': None,
            'solver_stats': sat_result.get('stats', {})
        }

        # SAT인 경우 생성행렬 복원 및 검증
        if sat_result['status'] == 'SAT':
            if verbose:
                print(f"\n{'='*60}")
                print(f"해 발견! 생성행렬 복원 중...")
                print(f"{'='*60}\n")

            # 생성행렬 복원
            G = self.parser.parse_sat_model(
                sat_result['model'],
                self.encoder.var_map
            )
            result['generator_matrix'] = G

            # 검증
            verification = self.parser.verify_generator_matrix(G, verbose=verbose)
            result['verification'] = verification

            # 결과 저장
            result_file = str(self.working_dir / f"{self.code_name}_result.txt")
            self.parser.export_results(G, result_file, include_codewords=True)

            # 생성행렬 NumPy 파일로 저장
            np_file = str(self.working_dir / f"{self.code_name}_matrix.npy")
            np.save(np_file, G)
            print(f"\n생성행렬 저장: {np_file}")

        elif sat_result['status'] == 'UNSAT':
            if verbose:
                print(f"\n{'='*60}")
                print(f"해가 존재하지 않습니다 (UNSAT)")
                print(f"{'='*60}\n")
                print(f"이는 [{self.n}, {self.k}, {self.d_min}] 코드가 존재하지 않음을 의미합니다.")

        elif sat_result['status'] == 'TIMEOUT':
            if verbose:
                print(f"\n{'='*60}")
                print(f"시간 초과 (TIMEOUT)")
                print(f"{'='*60}\n")
                print(f"제한 시간: {timeout}초")
                print(f"제안:")
                print(f"  1. 제한 시간을 늘리세요")
                print(f"  2. 더 강력한 대칭성 파괴 제약을 추가하세요")
                print(f"  3. 더 작은 파라미터로 시작하세요")

        result['total_time'] = time.time() - start_time

        if verbose:
            print(f"\n{'='*60}")
            print(f"총 실행 시간: {result['total_time']:.2f}초")
            print(f"{'='*60}\n")

        return result

    def search_codes_progressive(
        self,
        d_min_start: int,
        d_min_end: int,
        kissat_path: str = "kissat",
        timeout_per_instance: int = 300
    ) -> Dict[int, Dict]:
        """
        점진적 탐색: d_min을 start부터 end까지 증가시키며 탐색.

        Parameters
        ----------
        d_min_start : int
            시작 최소 거리
        d_min_end : int
            종료 최소 거리
        kissat_path : str
            Kissat 경로
        timeout_per_instance : int
            각 인스턴스당 제한 시간(초)

        Returns
        -------
        dict
            {d_min: result} 형태의 결과 딕셔너리
        """
        print(f"\n{'='*60}")
        print(f"점진적 탐색 시작")
        print(f"{'='*60}")
        print(f"[{self.n}, {self.k}] 코드")
        print(f"최소 거리 범위: {d_min_start} ~ {d_min_end}")
        print(f"인스턴스당 제한 시간: {timeout_per_instance}초")
        print(f"{'='*60}\n")

        results = {}

        for d in range(d_min_start, d_min_end + 1):
            print(f"\n{'#'*60}")
            print(f"# 현재 목표: d_min = {d}")
            print(f"{'#'*60}\n")

            # 새로운 인코더 생성
            self.d_min = d
            self.code_name = f"code_{self.n}_{self.k}_{d}"
            self.encoder = CNFEncoder(self.n, self.k, d, self.systematic)

            # 탐색
            result = self.solve(
                kissat_path=kissat_path,
                timeout=timeout_per_instance,
                verbose=True
            )

            results[d] = result

            # UNSAT이면 더 이상 증가시켜도 의미 없음
            if result['status'] == 'UNSAT':
                print(f"\nd_min = {d}에서 UNSAT. 탐색 종료.")
                break

        print(f"\n{'='*60}")
        print(f"점진적 탐색 완료")
        print(f"{'='*60}")
        print(f"\n결과 요약:")
        for d, res in results.items():
            status = res['status']
            symbol = '✓' if status == 'SAT' else '✗'
            print(f"  d_min = {d}: {symbol} {status}")

        return results

    def compare_with_singleton_bound(self) -> Dict:
        """
        Singleton Bound와 비교하여 코드의 최적성 평가.

        Singleton Bound: d_min <= n - k + 1

        Returns
        -------
        dict
            {
                'singleton_bound': int,
                'target_d_min': int,
                'achievable': bool,
                'optimal': bool
            }
        """
        singleton = self.n - self.k + 1

        result = {
            'singleton_bound': singleton,
            'target_d_min': self.d_min,
            'achievable': (self.d_min <= singleton),
            'optimal': False
        }

        print(f"\n{'='*60}")
        print(f"Singleton Bound 분석")
        print(f"{'='*60}")
        print(f"Singleton Bound: d_min <= n - k + 1 = {self.n} - {self.k} + 1 = {singleton}")
        print(f"목표 d_min: {self.d_min}")
        print(f"달성 가능: {result['achievable']}")

        if self.d_min == singleton:
            result['optimal'] = True
            print(f"\n✓ MDS (Maximum Distance Separable) 코드!")
            print(f"  이론적으로 최적의 코드입니다.")
        elif self.d_min < singleton:
            print(f"\n이론적으로 달성 가능한 범위 내입니다.")
        else:
            print(f"\n✗ Singleton Bound를 초과합니다.")
            print(f"  이러한 코드는 존재하지 않습니다.")

        print(f"{'='*60}\n")

        return result

    def load_existing_result(self, matrix_file: str) -> Dict:
        """
        이전에 저장된 생성행렬을 불러와 검증.

        Parameters
        ----------
        matrix_file : str
            .npy 형식의 행렬 파일 경로

        Returns
        -------
        dict
            검증 결과
        """
        print(f"기존 결과 로드 중: {matrix_file}")
        G = np.load(matrix_file)

        verification = self.parser.verify_generator_matrix(G, verbose=True)

        return {
            'generator_matrix': G,
            'verification': verification
        }

    def add_blocking_clause(self, model: Dict[int, bool]):
        """
        주어진 SAT model을 blocking clause로 추가하여 같은 해가 다시 나오지 않도록 함.

        Parameters
        ----------
        model : dict
            변수 번호 → True/False 매핑
        """
        # blocking clause: 최소 하나의 변수는 현재 값과 달라야 함
        # ¬x1 ∨ ¬x2 ∨ ... (모든 True 변수들)
        # x1 ∨ x2 ∨ ... (모든 False 변수들)

        clause = []
        for var, value in model.items():
            if value:
                clause.append(-var)  # True였던 변수는 False가 되어야
            else:
                clause.append(var)   # False였던 변수는 True가 되어야

        self.encoder.clauses.append(clause)

    def solve_multiple(
        self,
        solver_type: str = "kissat",
        solver_path: Optional[str] = None,
        num_solutions: int = 10,
        timeout_per_instance: Optional[int] = None,
        verbose: bool = True
    ) -> list:
        """
        여러 개의 서로 다른 생성행렬을 찾음 (inequivalent codes).

        Parameters
        ----------
        kissat_path : str
            Kissat 실행 파일 경로
        num_solutions : int
            찾을 해의 개수
        timeout_per_instance : int, optional
            각 인스턴스당 제한 시간(초)
        verbose : bool
            상세 출력 여부

        Returns
        -------
        list of dict
            각 해에 대한 결과 리스트
        """
        import time

        if verbose:
            print(f"\n{'='*70}")
            print(f"다중 해 탐색 시작")
            print(f"{'='*70}")
            print(f"목표: [{self.n}, {self.k}, {self.d_min}] 코드")
            print(f"찾을 해의 개수: {num_solutions}개")
            print(f"{'='*70}\n")

        all_results = []

        # 초기 CNF 생성
        cnf_file = self.generate_cnf()

        for i in range(num_solutions):
            if verbose:
                print(f"\n{'#'*70}")
                print(f"# 해 #{i+1}/{num_solutions} 탐색 중...")
                print(f"{'#'*70}\n")

            start_time = time.time()

            # SAT 솔버 초기화
            self.solver = SATSolver(solver_type=solver_type, solver_path=solver_path)

            # CNF를 임시 파일로 다시 저장 (blocking clause 포함)
            temp_cnf = str(self.working_dir / f"{self.code_name}_iter{i+1}.cnf")
            self.encoder.export_to_dimacs(temp_cnf)

            # SAT 문제 풀기
            sat_result = self.solver.solve(temp_cnf, timeout=timeout_per_instance, verbose=verbose)

            result = {
                'solution_number': i + 1,
                'status': sat_result['status'],
                'generator_matrix': None,
                'verification': None,
                'cnf_file': temp_cnf,
                'solver_time': sat_result['time'],
                'solver_stats': sat_result.get('stats', {})
            }

            # SAT인 경우 생성행렬 복원 및 검증
            if sat_result['status'] == 'SAT':
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"해 #{i+1} 발견! 생성행렬 복원 중...")
                    print(f"{'='*60}\n")

                # 생성행렬 복원
                G = self.parser.parse_sat_model(
                    sat_result['model'],
                    self.encoder.var_map
                )
                result['generator_matrix'] = G

                # 검증
                verification = self.parser.verify_generator_matrix(G, verbose=verbose)
                result['verification'] = verification

                # 결과 저장
                result_file = str(self.working_dir / f"{self.code_name}_solution{i+1}_result.txt")
                self.parser.export_results(G, result_file, include_codewords=False)

                # 생성행렬 NumPy 파일로 저장
                np_file = str(self.working_dir / f"{self.code_name}_solution{i+1}_matrix.npy")
                np.save(np_file, G)
                if verbose:
                    print(f"생성행렬 저장: {np_file}")

                # Blocking clause 추가하여 이 해가 다시 나오지 않도록
                self.add_blocking_clause(sat_result['model'])

                all_results.append(result)

            elif sat_result['status'] == 'UNSAT':
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"더 이상 해가 없습니다 (UNSAT)")
                    print(f"{'='*60}\n")
                    print(f"총 {i}개의 서로 다른 해를 찾았습니다.")
                break

            elif sat_result['status'] == 'TIMEOUT':
                if verbose:
                    print(f"\n{'='*60}")
                    print(f"시간 초과 (TIMEOUT)")
                    print(f"{'='*60}\n")
                    print(f"현재까지 {i}개의 해를 찾았습니다.")
                break

            result['elapsed_time'] = time.time() - start_time

        # 최종 요약
        if verbose:
            print(f"\n\n{'='*70}")
            print(f"다중 해 탐색 완료")
            print(f"{'='*70}")
            print(f"총 {len(all_results)}개의 서로 다른 생성행렬을 찾았습니다.\n")

            for i, res in enumerate(all_results, 1):
                print(f"해 #{i}:")
                print(f"  파일: {self.code_name}_solution{i}_matrix.npy")
                print(f"  최소 거리: {res['verification']['minimum_distance']}")
                print(f"  탐색 시간: {res['solver_time']:.2f}초")
                print()

        return all_results


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║       Binary Linear Code Generator with Kissat SAT Solver   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # 예제 1: [7, 4, 3] 해밍 코드 탐색
    print("\n" + "="*60)
    print("예제 1: [7, 4, 3] 해밍 코드 탐색")
    print("="*60)

    generator1 = CodeGenerator(n=7, k=4, d_min=3, systematic=True)

    # Singleton Bound 확인
    generator1.compare_with_singleton_bound()

    # CNF 생성
    cnf_file = generator1.generate_cnf()

    print(f"\nCNF 파일이 생성되었습니다: {cnf_file}")
    print("\nKissat 실행을 위해서는 다음과 같이 하세요:")
    print("  generator1.solve(kissat_path='./kissat/build/kissat')")

    # 예제 2: [16, 7, 6] 코드 - 목표 문제
    print("\n\n" + "="*60)
    print("예제 2: [16, 7, 6] 코드 탐색 준비")
    print("="*60)

    generator2 = CodeGenerator(n=16, k=7, d_min=6, systematic=True)

    # Singleton Bound 확인
    generator2.compare_with_singleton_bound()

    # CNF 생성
    cnf_file2 = generator2.generate_cnf()

    print(f"\nCNF 파일이 생성되었습니다: {cnf_file2}")
    print(f"파일 크기: {os.path.getsize(cnf_file2) / 1024:.2f} KB")

    print("\n" + "="*60)
    print("실행 방법:")
    print("="*60)
    print("""
# Kissat이 설치되어 있다면:
result = generator2.solve(kissat_path='./kissat/build/kissat', timeout=3600)

# 결과 확인:
if result['status'] == 'SAT':
    G = result['generator_matrix']
    print("생성행렬 발견!")
    print(G)
    """)
