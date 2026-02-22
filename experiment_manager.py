#!/usr/bin/env python3
"""
통합 실험 관리 스크립트

중단된 실험 재개, 모니터링, 관리를 하나의 인터페이스로 제공합니다.
"""

import sys
import os
from pathlib import Path

# 현재 디렉토리를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from experiment_checkpoint import ExperimentResumeManager, ExperimentCheckpoint
from experiment_monitor import ExperimentProgressTracker, KissatMonitor


class ExperimentManager:
    """
    실험 관리 통합 클래스
    """

    def __init__(self, working_dir: str = "./intermediate_results"):
        """
        Parameters
        ----------
        working_dir : str
            작업 디렉토리
        """
        self.working_dir = Path(working_dir)
        self.resume_manager = ExperimentResumeManager(str(self.working_dir))
        self.progress_tracker = ExperimentProgressTracker(str(self.working_dir))
        self.kissat_path = str(Path(__file__).parent / 'kissat' / 'build' / 'kissat')

    def show_main_menu(self):
        """메인 메뉴 표시"""
        print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║              실험 관리 시스템 (Experiment Manager)              ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

메뉴:
  1. 실행 중인 실험 상태 확인
  2. 중단된 실험 찾기 및 재개
  3. 실시간 모니터링 시작
  4. 실험 결과 요약
  5. 도움말
  q. 종료

        """)

    def check_running_status(self):
        """실행 중인 실험 상태 확인"""
        print("\n" + "="*80)
        print(" 실행 중인 실험 상태 확인")
        print("="*80)

        self.progress_tracker.print_all_running_status()

        input("\nEnter를 눌러 계속...")

    def find_and_resume(self):
        """중단된 실험 찾기 및 재개"""
        print("\n" + "="*80)
        print(" 중단된 실험 찾기 및 재개")
        print("="*80)

        # 중단된 실험 스캔
        experiments = self.resume_manager.scan_interrupted_experiments()

        if not experiments:
            print("\n중단된 실험이 없습니다.")
            print("모든 실험이 완료되었거나 시작되지 않았습니다.")
            input("\nEnter를 눌러 계속...")
            return

        # 중단된 실험 목록 출력
        self.resume_manager.print_interrupted_experiments(experiments)

        # 재개할 실험 선택
        selected = self.resume_manager.select_experiment_to_resume(experiments)

        if selected is None:
            print("\n취소되었습니다.")
            input("\nEnter를 눌러 계속...")
            return

        # 단일 실험 또는 전체 실험
        if isinstance(selected, dict):
            experiments_to_run = [selected]
        else:
            experiments_to_run = selected

        # 제한 시간 설정
        print("\n제한 시간 설정:")
        print("  Enter: 무제한")
        print("  초 단위로 입력 (예: 3600 = 1시간, 86400 = 24시간)")
        timeout_input = input("제한 시간 (초): ").strip()

        if timeout_input:
            try:
                timeout = int(timeout_input)
            except ValueError:
                print("잘못된 입력. 무제한으로 설정.")
                timeout = None
        else:
            timeout = None

        # 실험 재개
        results = []

        for i, exp in enumerate(experiments_to_run, 1):
            print(f"\n\n{'#'*80}")
            print(f"# 실험 {i}/{len(experiments_to_run)}")
            print(f"{'#'*80}")

            result = self.resume_manager.resume_experiment(
                experiment=exp,
                kissat_path=self.kissat_path,
                timeout=timeout,
                verbose=True
            )

            results.append(result)

            # SAT 발견 시 결과 처리
            if result['status'] == 'SAT':
                self._process_sat_result(exp, result)

        # 최종 요약
        self._print_resume_summary(results)

        input("\nEnter를 눌러 계속...")

    def _process_sat_result(self, experiment: dict, result: dict):
        """SAT 결과 처리 및 저장"""
        n, k, d_min = experiment['code_params']

        print(f"\n🎉 해 발견!")
        print(f"\n결과 파싱 및 저장...")

        try:
            from engines.result_parser import ResultParser
            import numpy as np

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
            result_file = experiment['working_dir'] / f"code_{n}_{k}_{d_min}_matrix.npy"
            np.save(result_file, G)
            print(f"\n저장: {result_file}")

            # 상세 정보 저장
            info_file = experiment['working_dir'] / f"code_{n}_{k}_{d_min}_result.txt"
            with open(info_file, 'w') as f:
                f.write(f"Binary Linear Code [{n}, {k}, {d_min}]\n")
                f.write(f"="*60 + "\n\n")
                f.write(f"검증:\n")
                f.write(f"  Rank: {verification['rank']}/{k}\n")
                f.write(f"  최소 거리: {verification['minimum_distance']}\n")
                f.write(f"  목표 달성: {verification['meets_requirement']}\n\n")
                f.write(f"생성행렬 G ({k}×{n}):\n")
                f.write(str(G) + "\n")

            print(f"상세 정보 저장: {info_file}")

        except Exception as e:
            print(f"결과 처리 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()

    def _print_resume_summary(self, results: list):
        """재개 실험 요약 출력"""
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

    def start_monitoring(self):
        """실시간 모니터링 시작"""
        print("\n" + "="*80)
        print(" 실시간 모니터링")
        print("="*80)

        print("\n모니터링 모드:")
        print("  1. 특정 실험 모니터링")
        print("  2. 모든 실험 모니터링")
        print("  q. 취소")

        choice = input("\n선택: ").strip().lower()

        if choice == 'q':
            return

        elif choice == '1':
            # 실행 중인 실험 목록
            running = self.progress_tracker.get_all_running_experiments()

            if not running:
                print("\n실행 중인 실험이 없습니다.")
                input("\nEnter를 눌러 계속...")
                return

            print("\n실행 중인 실험:")
            for i, exp in enumerate(running, 1):
                n, k, d_min = exp['code_params']
                print(f"  {i}. [{n}, {k}, {d_min}]")

            try:
                idx = int(input("\n모니터링할 실험 번호: ")) - 1
                if 0 <= idx < len(running):
                    exp = running[idx]
                    monitor = KissatMonitor(str(exp['output_file']))

                    interval = input("업데이트 간격 (초, 기본 10): ").strip()
                    interval = int(interval) if interval else 10

                    monitor.watch(interval=interval)
                else:
                    print("잘못된 번호입니다.")
            except ValueError:
                print("잘못된 입력입니다.")
            except KeyboardInterrupt:
                print("\n모니터링 중단됨.")

        elif choice == '2':
            interval = input("업데이트 간격 (초, 기본 30): ").strip()
            interval = int(interval) if interval else 30

            try:
                self.progress_tracker.watch_all(interval=interval)
            except KeyboardInterrupt:
                print("\n모니터링 중단됨.")

        input("\nEnter를 눌러 계속...")

    def show_results_summary(self):
        """실험 결과 요약"""
        print("\n" + "="*80)
        print(" 실험 결과 요약")
        print("="*80)

        if not self.working_dir.exists():
            print("\n결과 디렉토리가 없습니다.")
            input("\nEnter를 눌러 계속...")
            return

        completed_experiments = []

        for dir_path in self.working_dir.iterdir():
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

                # 결과 파일 확인
                result_file = dir_path / f"{dir_name}_matrix.npy"

                if result_file.exists():
                    import numpy as np
                    G = np.load(result_file)

                    completed_experiments.append({
                        'code_params': (n, k, d_min),
                        'result_file': result_file,
                        'matrix_shape': G.shape,
                        'found_time': result_file.stat().st_mtime
                    })

            except Exception:
                continue

        if not completed_experiments:
            print("\n완료된 실험이 없습니다.")
        else:
            print(f"\n완료된 실험: {len(completed_experiments)}개\n")
            print(f"{'코드':15s} {'행렬 크기':15s} {'완료 시각':20s}")
            print("-"*80)

            for exp in completed_experiments:
                n, k, d_min = exp['code_params']
                code_str = f"[{n}, {k}, {d_min}]"
                shape_str = f"{exp['matrix_shape'][0]}×{exp['matrix_shape'][1]}"

                from datetime import datetime
                time_str = datetime.fromtimestamp(exp['found_time']).strftime('%Y-%m-%d %H:%M:%S')

                print(f"✓ {code_str:13s} {shape_str:15s} {time_str:20s}")

        input("\nEnter를 눌러 계속...")

    def show_help(self):
        """도움말 표시"""
        print("""
╔══════════════════════════════════════════════════════════════════╗
║                          도움말                                  ║
╚══════════════════════════════════════════════════════════════════╝

1. 실행 중인 실험 상태 확인
   - 현재 Kissat 프로세스가 실행 중인 실험 목록 확인
   - Conflicts, Restarts, 메모리 사용량 등 통계 표시
   - 처리 속도 (conflicts/sec) 계산

2. 중단된 실험 찾기 및 재개
   - intermediate_results 디렉토리 스캔
   - CNF 파일은 있지만 결과가 없는 실험 탐지
   - 선택한 실험을 재개하고 결과 저장

3. 실시간 모니터링 시작
   - 특정 실험: 선택한 실험의 진행 상황을 주기적으로 업데이트
   - 모든 실험: 실행 중인 모든 실험을 한 화면에서 모니터링

4. 실험 결과 요약
   - 완료된 실험 목록 (*.npy 파일이 있는 실험)
   - 생성행렬 크기 및 완료 시각 표시

디렉토리 구조:
  intermediate_results/
  ├── code_35_10_13/
  │   ├── code_35_10_13.cnf          # CNF 파일
  │   ├── code_35_10_13.cnf.out      # Kissat 출력
  │   ├── code_35_10_13_matrix.npy   # 생성행렬 (완료 시)
  │   └── code_35_10_13_result.txt   # 상세 결과 (완료 시)
  └── ...

실험 상태:
  - CNF_READY: CNF 파일만 있음 (시작 전)
  - RUNNING: Kissat 실행 중
  - INTERRUPTED: 중단됨 (재개 필요)
  - SAT: 해 발견 (완료)
  - UNSAT: 해 없음 (완료)
  - TIMEOUT: 시간 초과

팁:
  - 장시간 실험은 screen이나 tmux 사용 권장
  - 제한 시간을 설정하면 자동으로 중단됨
  - 중단된 실험은 언제든 재개 가능
  - Kissat은 incremental solving 미지원 (재개 시 새로 시작)

관련 파일:
  - experiment_manager.py: 이 통합 스크립트
  - experiment_checkpoint.py: 재개 시스템
  - experiment_monitor.py: 모니터링 시스템
  - RESUME_GUIDE.md: 상세 사용 가이드

        """)

        input("\nEnter를 눌러 계속...")

    def run(self):
        """메인 루프"""
        while True:
            # 화면 지우기
            os.system('clear' if os.name == 'posix' else 'cls')

            self.show_main_menu()

            choice = input("선택: ").strip().lower()

            if choice == '1':
                self.check_running_status()
            elif choice == '2':
                self.find_and_resume()
            elif choice == '3':
                self.start_monitoring()
            elif choice == '4':
                self.show_results_summary()
            elif choice == '5':
                self.show_help()
            elif choice == 'q':
                print("\n종료합니다.")
                break
            else:
                print("\n잘못된 선택입니다.")
                input("\nEnter를 눌러 계속...")


def main():
    """메인 함수"""
    manager = ExperimentManager(working_dir="./intermediate_results")

    try:
        manager.run()
    except KeyboardInterrupt:
        print("\n\n프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
