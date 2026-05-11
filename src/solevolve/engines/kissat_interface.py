"""
Kissat SAT Solver Interface

이 모듈은 Kissat 솔버와의 인터페이스를 제공합니다.
"""

import subprocess
import os
import time
from typing import Dict, Optional, List
from pathlib import Path


class KissatSolver:
    """
    Kissat SAT 솔버 래퍼 클래스.
    """

    def __init__(self, kissat_path: str = "kissat"):
        """
        Parameters
        ----------
        kissat_path : str
            Kissat 실행 파일 경로 (기본값: PATH에서 'kissat' 탐색)
        """
        self.kissat_path = kissat_path
        self._verify_kissat()

    def _verify_kissat(self):
        """Kissat이 설치되어 있고 실행 가능한지 확인."""
        try:
            result = subprocess.run(
                [self.kissat_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version_info = result.stdout.strip()
                print(f"Kissat 발견: {version_info}")
            else:
                raise RuntimeError("Kissat 실행 실패")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Kissat 실행 파일을 찾을 수 없습니다: {self.kissat_path}\n"
                "Kissat을 설치하세요:\n"
                "  git clone https://github.com/arminbiere/kissat.git\n"
                "  cd kissat\n"
                "  ./configure && make\n"
                "  # 그 후 kissat_path에 build/kissat 경로 지정"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Kissat 버전 확인 시간 초과")

    def solve(
        self,
        cnf_file: str,
        output_file: Optional[str] = None,
        timeout: Optional[int] = None,
        verbose: bool = True
    ) -> Dict:
        """
        CNF 파일에 대해 Kissat 솔버 실행.

        Parameters
        ----------
        cnf_file : str
            입력 DIMACS CNF 파일 경로
        output_file : str, optional
            출력 결과 파일 경로 (기본값: cnf_file.out)
        timeout : int, optional
            제한 시간(초), None이면 무제한
        verbose : bool
            진행 상황 출력 여부

        Returns
        -------
        dict
            {
                'status': 'SAT' | 'UNSAT' | 'TIMEOUT' | 'ERROR',
                'time': float (실행 시간, 초),
                'model': Dict[int, bool] | None (SAT인 경우 변수 할당),
                'output_file': str (결과 파일 경로),
                'stats': Dict (통계 정보)
            }
        """
        if not os.path.exists(cnf_file):
            raise FileNotFoundError(f"CNF 파일이 존재하지 않습니다: {cnf_file}")

        if output_file is None:
            output_file = cnf_file + ".out"

        if verbose:
            print(f"\n{'='*60}")
            print(f"Kissat 솔버 실행")
            print(f"{'='*60}")
            print(f"입력 파일: {cnf_file}")
            print(f"출력 파일: {output_file}")
            if timeout:
                print(f"제한 시간: {timeout}초")
            print()

        start_time = time.time()

        try:
            # Kissat 실행
            with open(output_file, 'w') as f_out:
                result = subprocess.run(
                    [self.kissat_path, cnf_file],
                    stdout=f_out,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout
                )

            elapsed_time = time.time() - start_time

            # 결과 파싱
            status, model, stats = self._parse_output(output_file)

            if verbose:
                print(f"상태: {status}")
                print(f"실행 시간: {elapsed_time:.2f}초")
                if stats:
                    print("\n통계:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")

            return {
                'status': status,
                'time': elapsed_time,
                'model': model,
                'output_file': output_file,
                'stats': stats
            }

        except subprocess.TimeoutExpired:
            elapsed_time = time.time() - start_time
            if verbose:
                print(f"시간 초과! ({timeout}초)")

            return {
                'status': 'TIMEOUT',
                'time': elapsed_time,
                'model': None,
                'output_file': output_file,
                'stats': {}
            }

        except Exception as e:
            elapsed_time = time.time() - start_time
            if verbose:
                print(f"오류 발생: {str(e)}")

            return {
                'status': 'ERROR',
                'time': elapsed_time,
                'model': None,
                'output_file': output_file,
                'stats': {},
                'error': str(e)
            }

    def _parse_output(self, output_file: str) -> tuple:
        """
        Kissat 출력 파일 파싱.

        Returns
        -------
        tuple
            (status, model, stats)
            - status: 'SAT' | 'UNSAT' | 'UNKNOWN'
            - model: Dict[int, bool] | None
            - stats: Dict[str, any]
        """
        status = 'UNKNOWN'
        model = None
        stats = {}

        with open(output_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            # 상태 라인: s SATISFIABLE | s UNSATISFIABLE
            if line.startswith('s '):
                if 'SATISFIABLE' in line and 'UNSATISFIABLE' not in line:
                    status = 'SAT'
                elif 'UNSATISFIABLE' in line:
                    status = 'UNSAT'

            # 모델 라인: v <리터럴들> 0
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

            # 통계 라인: c <키>: <값>
            elif line.startswith('c '):
                parts = line[2:].split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value_str = parts[1].strip()
                    # 숫자로 변환 시도
                    try:
                        if '.' in value_str:
                            value = float(value_str)
                        else:
                            value = int(value_str)
                    except ValueError:
                        value = value_str
                    stats[key] = value

        return status, model, stats

    def solve_incremental(
        self,
        cnf_files: List[str],
        timeout_per_instance: Optional[int] = None,
        verbose: bool = True
    ) -> List[Dict]:
        """
        여러 CNF 파일을 순차적으로 풀기.

        Parameters
        ----------
        cnf_files : List[str]
            CNF 파일 경로 리스트
        timeout_per_instance : int, optional
            각 인스턴스당 제한 시간(초)
        verbose : bool
            진행 상황 출력 여부

        Returns
        -------
        List[Dict]
            각 파일에 대한 결과 딕셔너리 리스트
        """
        results = []

        for i, cnf_file in enumerate(cnf_files, 1):
            if verbose:
                print(f"\n진행 상황: {i}/{len(cnf_files)}")

            result = self.solve(
                cnf_file=cnf_file,
                timeout=timeout_per_instance,
                verbose=verbose
            )
            results.append(result)

            # SAT 발견 시 조기 종료 (선택적)
            if result['status'] == 'SAT':
                if verbose:
                    print(f"\n해 발견! 나머지 {len(cnf_files) - i}개 파일 건너뜀.")
                break

        return results

    @staticmethod
    def install_kissat(target_dir: str = "./kissat"):
        """
        Kissat 자동 설치 스크립트.

        Parameters
        ----------
        target_dir : str
            Kissat을 설치할 디렉토리
        """
        target_path = Path(target_dir)

        print(f"Kissat 설치 시작: {target_path.absolute()}")

        if target_path.exists():
            print(f"경고: {target_path}가 이미 존재합니다.")
            response = input("덮어쓰시겠습니까? (y/n): ")
            if response.lower() != 'y':
                print("설치 취소됨.")
                return

        try:
            # Git clone
            print("Git clone 중...")
            subprocess.run(
                ["git", "clone", "https://github.com/arminbiere/kissat.git", str(target_path)],
                check=True
            )

            # Configure
            print("Configure 중...")
            subprocess.run(
                ["./configure"],
                cwd=target_path,
                check=True
            )

            # Make
            print("Make 중...")
            subprocess.run(
                ["make"],
                cwd=target_path,
                check=True
            )

            kissat_binary = target_path / "build" / "kissat"
            if kissat_binary.exists():
                print(f"\n✓ Kissat 설치 성공!")
                print(f"  실행 파일: {kissat_binary.absolute()}")
                print(f"\n사용 예:")
                print(f"  solver = KissatSolver(kissat_path='{kissat_binary}')")
            else:
                print("\n✗ 빌드 파일을 찾을 수 없습니다.")

        except subprocess.CalledProcessError as e:
            print(f"\n✗ 설치 실패: {e}")
        except Exception as e:
            print(f"\n✗ 오류 발생: {e}")


if __name__ == "__main__":
    # 예제 사용법
    print("Kissat Interface 모듈 테스트\n")

    # Kissat 설치 여부 확인
    try:
        solver = KissatSolver(kissat_path="kissat")
        print("✓ Kissat이 PATH에 설치되어 있습니다.\n")
    except FileNotFoundError:
        print("✗ Kissat이 PATH에 없습니다.")
        print("자동 설치를 시작하려면 다음을 실행하세요:")
        print("  KissatSolver.install_kissat()\n")

        # 자동 설치 시도 (주석 해제하여 사용)
        # KissatSolver.install_kissat()

    # 예제 CNF 파일이 있다면 테스트
    test_cnf = "hamming_7_4_3.cnf"
    if os.path.exists(test_cnf):
        print(f"테스트 CNF 파일 발견: {test_cnf}")
        solver = KissatSolver()
        result = solver.solve(test_cnf, timeout=60)
        print(f"\n결과: {result['status']}")
        if result['model']:
            print(f"모델 크기: {len(result['model'])} 변수")
