#!/usr/bin/env python3
"""
실험 체크포인트 및 재개 시스템

중단된 실험을 저장하고 재개할 수 있는 기능을 제공합니다.
"""

import json
import os
import time
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import subprocess


class ExperimentCheckpoint:
    """
    실험 상태를 저장하고 복원하는 체크포인트 시스템
    """

    def __init__(self, working_dir: str = "./intermediate_results"):
        """
        Parameters
        ----------
        working_dir : str
            체크포인트를 저장할 작업 디렉토리
        """
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.working_dir / "experiment_checkpoint.json"

    def save_checkpoint(self,
                       experiment_params: Dict,
                       current_status: str,
                       elapsed_time: float,
                       additional_info: Optional[Dict] = None) -> None:
        """
        현재 실험 상태를 체크포인트로 저장

        Parameters
        ----------
        experiment_params : Dict
            실험 파라미터 (n, k, d_min 등)
        current_status : str
            현재 상태 ('CNF_GENERATED', 'SOLVING', 'INTERRUPTED', 'COMPLETED')
        elapsed_time : float
            경과 시간 (초)
        additional_info : Dict, optional
            추가 정보 (통계, 진행률 등)
        """
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'experiment_params': experiment_params,
            'current_status': current_status,
            'elapsed_time': elapsed_time,
            'additional_info': additional_info or {}
        }

        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        print(f"✓ 체크포인트 저장: {self.checkpoint_file}")

    def load_checkpoint(self) -> Optional[Dict]:
        """
        저장된 체크포인트를 로드

        Returns
        -------
        Dict or None
            체크포인트 데이터, 없으면 None
        """
        if not self.checkpoint_file.exists():
            return None

        with open(self.checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)

        print(f"✓ 체크포인트 로드: {self.checkpoint_file}")
        return checkpoint_data

    def has_checkpoint(self) -> bool:
        """체크포인트가 존재하는지 확인"""
        return self.checkpoint_file.exists()

    def clear_checkpoint(self) -> None:
        """체크포인트 삭제"""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            print(f"✓ 체크포인트 삭제: {self.checkpoint_file}")


class ExperimentResumeManager:
    """
    중단된 실험을 탐지하고 재개하는 매니저
    """

    def __init__(self, intermediate_results_dir: str = "./intermediate_results"):
        """
        Parameters
        ----------
        intermediate_results_dir : str
            중간 결과가 저장된 디렉토리
        """
        self.intermediate_results_dir = Path(intermediate_results_dir)
        self.checkpoint_manager = ExperimentCheckpoint(intermediate_results_dir)

    def scan_interrupted_experiments(self) -> List[Dict]:
        """
        중단된 실험들을 스캔하여 목록 반환

        Returns
        -------
        List[Dict]
            중단된 실험 정보 리스트
            각 항목: {
                'code_params': (n, k, d_min),
                'working_dir': Path,
                'cnf_file': Path,
                'output_file': Path,
                'status': str,
                'has_output': bool,
                'output_size': int,
                'cnf_size': int,
                'estimated_progress': float
            }
        """
        interrupted = []

        if not self.intermediate_results_dir.exists():
            return interrupted

        # code_*_*_* 형태의 디렉토리 찾기
        for dir_path in self.intermediate_results_dir.iterdir():
            if not dir_path.is_dir():
                continue

            # code_n_k_d 형태 파싱
            dir_name = dir_path.name
            if not dir_name.startswith('code_'):
                continue

            try:
                parts = dir_name.split('_')
                if len(parts) < 4:
                    continue

                n = int(parts[1])
                k = int(parts[2])
                d_min = int(parts[3])

                # CNF 파일 확인
                cnf_file = dir_path / f"{dir_name}.cnf"
                output_file = dir_path / f"{dir_name}.cnf.out"

                if not cnf_file.exists():
                    continue

                # 결과 파일(matrix.npy) 확인
                result_file = dir_path / f"{dir_name}_matrix.npy"

                # 이미 완료된 실험은 제외
                if result_file.exists():
                    continue

                # 출력 파일 상태 확인
                has_output = output_file.exists()
                output_size = output_file.stat().st_size if has_output else 0
                cnf_size = cnf_file.stat().st_size

                # 상태 판단
                if has_output and output_size > 0:
                    status = self._check_solving_status(output_file)
                    estimated_progress = self._estimate_progress(output_file)
                else:
                    status = 'CNF_READY'
                    estimated_progress = 0.0

                interrupted.append({
                    'code_params': (n, k, d_min),
                    'working_dir': dir_path,
                    'cnf_file': cnf_file,
                    'output_file': output_file,
                    'status': status,
                    'has_output': has_output,
                    'output_size': output_size,
                    'cnf_size': cnf_size,
                    'estimated_progress': estimated_progress
                })

            except (ValueError, IndexError):
                continue

        return interrupted

    def _check_solving_status(self, output_file: Path) -> str:
        """
        Kissat 출력 파일에서 현재 상태 확인

        Returns
        -------
        str
            'RUNNING', 'COMPLETED', 'TIMEOUT', 'ERROR'
        """
        try:
            with open(output_file, 'r') as f:
                lines = f.readlines()

            # 마지막 몇 줄 확인
            for line in reversed(lines[-50:]):
                line = line.strip()

                if line.startswith('s '):
                    if 'SATISFIABLE' in line and 'UNSATISFIABLE' not in line:
                        return 'SAT'
                    elif 'UNSATISFIABLE' in line:
                        return 'UNSAT'

            # SAT/UNSAT 결과가 없으면 아직 실행 중이거나 중단됨
            return 'INTERRUPTED'

        except Exception as e:
            print(f"출력 파일 읽기 오류: {e}")
            return 'ERROR'

    def _estimate_progress(self, output_file: Path) -> float:
        """
        Kissat 출력에서 진행률 추정

        Kissat은 conflicts 수를 출력하므로 이를 기반으로 진행률 추정
        (정확한 진행률은 알 수 없지만 대략적인 활동량 표시)

        Returns
        -------
        float
            추정 진행률 (0.0 ~ 1.0), 정확하지 않으므로 참고용
        """
        try:
            with open(output_file, 'r') as f:
                lines = f.readlines()

            # conflicts 수 찾기
            max_conflicts = 0
            for line in reversed(lines[-100:]):
                if line.startswith('c -') or line.startswith('c '):
                    parts = line.split()
                    # Kissat 통계 라인에서 숫자 찾기
                    for part in parts:
                        try:
                            num = int(part)
                            if num > max_conflicts:
                                max_conflicts = num
                        except ValueError:
                            continue

            # conflicts 기반 진행률 (임의의 스케일)
            # 실제로는 문제마다 다르므로 참고용
            if max_conflicts > 0:
                # 로그 스케일로 0~1 사이로 매핑
                import math
                progress = min(1.0, math.log10(max_conflicts + 1) / 10.0)
                return round(progress, 3)
            else:
                return 0.0

        except Exception:
            return 0.0

    def get_kissat_process_info(self, code_name: str) -> Optional[Dict]:
        """
        실행 중인 Kissat 프로세스 정보 조회

        Parameters
        ----------
        code_name : str
            코드 이름 (예: 'code_35_10_13')

        Returns
        -------
        Dict or None
            프로세스 정보 {pid, cpu, memory, elapsed_time}
        """
        try:
            # ps 명령어로 kissat 프로세스 찾기
            result = subprocess.run(
                ['ps', 'aux'],
                capture_output=True,
                text=True,
                timeout=5
            )

            for line in result.stdout.split('\n'):
                if 'kissat' in line and code_name in line:
                    parts = line.split()
                    if len(parts) >= 11:
                        return {
                            'pid': int(parts[1]),
                            'cpu': parts[2],
                            'memory': parts[3],
                            'elapsed_time': parts[9],
                            'command': ' '.join(parts[10:])
                        }

            return None

        except Exception as e:
            print(f"프로세스 정보 조회 오류: {e}")
            return None

    def print_interrupted_experiments(self, experiments: List[Dict]) -> None:
        """
        중단된 실험 목록을 보기 좋게 출력

        Parameters
        ----------
        experiments : List[Dict]
            scan_interrupted_experiments() 결과
        """
        if not experiments:
            print("\n중단된 실험이 없습니다.")
            return

        print("\n" + "="*80)
        print(f" 중단된 실험 목록 ({len(experiments)}개)")
        print("="*80)

        print(f"\n{'번호':<4} {'코드':<15} {'상태':<12} {'CNF 크기':<12} {'출력 크기':<12} {'진행률':<8}")
        print("-"*80)

        for i, exp in enumerate(experiments, 1):
            n, k, d_min = exp['code_params']
            code_str = f"[{n},{k},{d_min}]"
            status = exp['status']
            cnf_size = f"{exp['cnf_size']/1024/1024:.1f} MB"
            output_size = f"{exp['output_size']/1024/1024:.1f} MB" if exp['has_output'] else "없음"
            progress = f"{exp['estimated_progress']*100:.0f}%" if exp['has_output'] else "0%"

            # 프로세스 실행 중 확인
            process_info = self.get_kissat_process_info(f"code_{n}_{k}_{d_min}")
            if process_info:
                status += " 🔄"

            print(f"{i:<4} {code_str:<15} {status:<12} {cnf_size:<12} {output_size:<12} {progress:<8}")

        print("-"*80)

    def select_experiment_to_resume(self, experiments: List[Dict]) -> Optional[Dict]:
        """
        사용자가 재개할 실험 선택

        Parameters
        ----------
        experiments : List[Dict]
            중단된 실험 목록

        Returns
        -------
        Dict or None
            선택된 실험 정보
        """
        if not experiments:
            return None

        print("\n재개할 실험 선택:")
        print("  번호 입력: 해당 실험 재개")
        print("  'a': 모든 실험 순차 재개")
        print("  'q': 취소")

        choice = input("\n선택: ").strip().lower()

        if choice == 'q':
            return None
        elif choice == 'a':
            return experiments  # 전체 리스트 반환
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(experiments):
                    return experiments[idx]
                else:
                    print("잘못된 번호입니다.")
                    return None
            except ValueError:
                print("잘못된 입력입니다.")
                return None

    def resume_experiment(self,
                         experiment: Dict,
                         kissat_path: str,
                         timeout: Optional[int] = None,
                         verbose: bool = True) -> Dict:
        """
        중단된 실험 재개

        Parameters
        ----------
        experiment : Dict
            재개할 실험 정보
        kissat_path : str
            Kissat 실행 파일 경로
        timeout : int, optional
            제한 시간 (초)
        verbose : bool
            진행 상황 출력 여부

        Returns
        -------
        Dict
            실험 결과
        """
        n, k, d_min = experiment['code_params']
        cnf_file = experiment['cnf_file']
        output_file = experiment['output_file']

        if verbose:
            print(f"\n{'='*80}")
            print(f" [{n}, {k}, {d_min}] 실험 재개")
            print(f"{'='*80}")
            print(f"CNF 파일: {cnf_file}")
            print(f"출력 파일: {output_file}")
            if experiment['has_output']:
                print(f"기존 출력 크기: {experiment['output_size']/1024/1024:.2f} MB")
                print(f"추정 진행률: {experiment['estimated_progress']*100:.0f}%")
            print()

        # Kissat 재실행
        # 기존 출력은 백업하고 새로 시작
        if output_file.exists():
            backup_file = output_file.with_suffix('.out.backup')
            output_file.rename(backup_file)
            if verbose:
                print(f"기존 출력 백업: {backup_file}")

        from engines.sat_solver_interface import SATSolver

        solver = SATSolver(solver_type="kissat", solver_path=kissat_path)

        start_time = time.time()
        result = solver.solve(
            cnf_file=str(cnf_file),
            output_file=str(output_file),
            timeout=timeout,
            verbose=verbose
        )
        elapsed_time = time.time() - start_time

        result['experiment_params'] = {'n': n, 'k': k, 'd_min': d_min}
        result['elapsed_time'] = elapsed_time

        return result


def main():
    """메인 함수 - 중단된 실험 찾기 및 재개"""

    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║          중단된 실험 재개 시스템 (Experiment Resume)            ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    # 중단�� 실험 스캔
    manager = ExperimentResumeManager(intermediate_results_dir="./intermediate_results")
    experiments = manager.scan_interrupted_experiments()

    if not experiments:
        print("\n중단된 실험이 없습니다.")
        print("모든 실험이 완료되었거나 시작되지 않았습니다.")
        return

    # 중단된 실험 목록 출력
    manager.print_interrupted_experiments(experiments)

    # 재개할 실험 선택
    selected = manager.select_experiment_to_resume(experiments)

    if selected is None:
        print("\n취소되었습니다.")
        return

    # Kissat 경로 설정
    KISSAT_PATH = '/Users/baegjaehyeon/CodeEvolve/src/kissat_code_search/kissat/build/kissat'

    # 단일 실험 재개
    if isinstance(selected, dict):
        experiments_to_run = [selected]
    else:  # 전체 실험 재개
        experiments_to_run = selected

    # 제한 시간 설정
    print("\n제한 시간 설정:")
    print("  기본값을 사용하려면 Enter")
    print("  초 단위로 입력 (예: 3600)")
    timeout_input = input("제한 시간 (초): ").strip()

    if timeout_input:
        try:
            timeout = int(timeout_input)
        except ValueError:
            print("잘못된 입력. 기본값 사용.")
            timeout = None
    else:
        timeout = None

    # 실험 재개
    results = []

    for i, exp in enumerate(experiments_to_run, 1):
        print(f"\n\n{'#'*80}")
        print(f"# 실험 {i}/{len(experiments_to_run)}")
        print(f"{'#'*80}")

        result = manager.resume_experiment(
            experiment=exp,
            kissat_path=KISSAT_PATH,
            timeout=timeout,
            verbose=True
        )

        results.append(result)

        # SAT 발견 시 결과 처리
        if result['status'] == 'SAT':
            n, k, d_min = exp['code_params']

            print(f"\n🎉 해 발견!")
            print(f"\n결과 파싱 및 저장...")

            # ResultParser로 결과 처리
            from engines.result_parser import ResultParser

            parser = ResultParser(n=n, k=k, d_min=d_min, systematic=True)

            # 모델에서 생성행렬 복원
            G = parser.extract_generator_matrix(result['model'])

            # 검증
            verification = parser.verify_code(G, d_min)

            print(f"\n검증 결과:")
            print(f"  Rank: {verification['rank']}/{k}")
            print(f"  최소 거리: {verification['minimum_distance']}")
            print(f"  목표 달성: {verification['meets_requirement']}")

            # 저장
            result_file = exp['working_dir'] / f"code_{n}_{k}_{d_min}_matrix.npy"
            import numpy as np
            np.save(result_file, G)
            print(f"\n저장: {result_file}")

    # 최종 요약
    print(f"\n\n" + "="*80)
    print(" 재개 실험 최종 요약")
    print("="*80)

    print(f"\n{'코드':15s} {'상태':12s} {'시간':15s}")
    print("-"*80)

    for result in results:
        params = result['experiment_params']
        code_str = f"[{params['n']},{params['k']},{params['d_min']}]"
        status = result['status']
        elapsed = result.get('elapsed_time', 0)

        if elapsed < 60:
            time_str = f"{elapsed:.2f}초"
        elif elapsed < 3600:
            time_str = f"{elapsed/60:.1f}분"
        else:
            time_str = f"{elapsed/3600:.1f}시간"

        symbol = "✓" if status == "SAT" else "✗"
        print(f"{symbol} {code_str:13s} {status:12s} {time_str:15s}")

    found = sum(1 for r in results if r['status'] == 'SAT')
    print(f"\n성공: {found}/{len(results)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
