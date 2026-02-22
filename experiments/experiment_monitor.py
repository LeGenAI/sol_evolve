#!/usr/bin/env python3
"""
실험 모니터링 시스템

실행 중인 Kissat 프로세스를 모니터링하고 진행 상황을 추적합니다.
"""

import os
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import re


class KissatMonitor:
    """
    실행 중인 Kissat 프로세스 모니터링
    """

    def __init__(self, output_file: str):
        """
        Parameters
        ----------
        output_file : str
            Kissat 출력 파일 경로
        """
        self.output_file = Path(output_file)
        self.start_time = None
        self.last_position = 0
        self.stats_history = []

    def get_current_stats(self) -> Optional[Dict]:
        """
        현재 통계 정보 파싱

        Returns
        -------
        Dict or None
            {
                'conflicts': int,
                'decisions': int,
                'propagations': int,
                'restarts': int,
                'reductions': int,
                'level': int,
                'variables': int,
                'clauses': int
            }
        """
        if not self.output_file.exists():
            return None

        try:
            with open(self.output_file, 'r') as f:
                # 파일의 마지막 부분만 읽기 (효율성)
                f.seek(0, 2)  # 파일 끝으로
                file_size = f.tell()

                # 마지막 10KB만 읽기
                read_size = min(10240, file_size)
                f.seek(max(0, file_size - read_size))
                tail_content = f.read()

            # Kissat 통계 라인 파싱
            # c - 시간 레벨 리스타트 ... conflicts ...
            stats_lines = [line for line in tail_content.split('\n') if line.startswith('c -')]

            if not stats_lines:
                return None

            # 마지막 통계 라인
            last_line = stats_lines[-1]
            parts = last_line.split()

            # Kissat 출력 형식 파싱
            # c - <seconds> <MB> <level> <restarts> <conflicts> <decisions> ...
            if len(parts) >= 10:
                try:
                    stats = {
                        'seconds': float(parts[2]),
                        'memory_mb': float(parts[3]),
                        'level': int(parts[4]),
                        'restarts': int(parts[5]),
                        'conflicts': int(parts[6]),
                        # 추가 정보는 Kissat 버전에 따라 다를 수 있음
                    }
                    return stats
                except (ValueError, IndexError):
                    pass

            return None

        except Exception as e:
            print(f"통계 파싱 오류: {e}")
            return None

    def get_progress_info(self) -> Dict:
        """
        진행 상황 정보 수집

        Returns
        -------
        Dict
            {
                'is_running': bool,
                'elapsed_time': float,
                'current_stats': Dict,
                'rate': Dict,  # conflicts/sec, decisions/sec 등
                'estimated_completion': str or None
            }
        """
        stats = self.get_current_stats()

        if stats is None:
            return {
                'is_running': False,
                'elapsed_time': 0,
                'current_stats': None,
                'rate': None,
                'estimated_completion': None
            }

        # 경과 시간
        elapsed_time = stats.get('seconds', 0)

        # 처리 속도 계산
        rate = {}
        if elapsed_time > 0:
            rate['conflicts_per_sec'] = stats.get('conflicts', 0) / elapsed_time
            rate['restarts_per_min'] = stats.get('restarts', 0) / (elapsed_time / 60)

        # 프로세스 실행 여부 확인
        is_running = self._is_process_running()

        return {
            'is_running': is_running,
            'elapsed_time': elapsed_time,
            'current_stats': stats,
            'rate': rate,
            'estimated_completion': None  # SAT 문제는 완료 시간 예측 불가능
        }

    def _is_process_running(self) -> bool:
        """Kissat 프로세스가 실행 중인지 확인"""
        try:
            # 출력 파일 이름에서 코드 이름 추출
            code_name = self.output_file.stem.replace('.cnf', '')

            result = subprocess.run(
                ['ps', 'aux'],
                capture_output=True,
                text=True,
                timeout=2
            )

            for line in result.stdout.split('\n'):
                if 'kissat' in line and code_name in line:
                    return True

            return False

        except Exception:
            return False

    def watch(self, interval: int = 10, duration: Optional[int] = None) -> None:
        """
        실시간 모니터링

        Parameters
        ----------
        interval : int
            업데이트 간격 (초)
        duration : int, optional
            모니터링 지속 시간 (초), None이면 무제한
        """
        print(f"\n{'='*80}")
        print(f" Kissat 실시간 모니터링")
        print(f"{'='*80}")
        print(f"출력 파일: {self.output_file}")
        print(f"업데이트 간격: {interval}초")
        print(f"Ctrl+C로 종료")
        print()

        start_watch = time.time()
        previous_stats = None

        try:
            while True:
                # 진행 정보 수집
                progress = self.get_progress_info()

                # 헤더 출력
                current_time = datetime.now().strftime('%H:%M:%S')
                print(f"\n[{current_time}] 상태 업데이트")
                print("-" * 80)

                if not progress['is_running']:
                    print("⚠️  Kissat 프로세스가 실행 중이지 않습니다.")

                    # 최종 상태 확인
                    if self._check_final_status():
                        print("✓ 실험이 완료되었습니다.")
                        break
                    else:
                        print("실험이 중단되었거나 아직 시작되지 않았습니다.")

                    time.sleep(interval)
                    continue

                # 통계 출력
                stats = progress['current_stats']
                if stats:
                    elapsed = timedelta(seconds=int(progress['elapsed_time']))

                    print(f"경과 시간: {elapsed}")
                    print(f"메모리 사용: {stats['memory_mb']:.1f} MB")
                    print(f"Conflicts: {stats['conflicts']:,}")
                    print(f"Restarts: {stats['restarts']:,}")
                    print(f"Level: {stats['level']}")

                    # 속도 계산
                    rate = progress['rate']
                    if rate:
                        print(f"\n처리 속도:")
                        print(f"  Conflicts/sec: {rate['conflicts_per_sec']:,.0f}")
                        print(f"  Restarts/min: {rate['restarts_per_min']:.1f}")

                    # 이전 통계와 비교
                    if previous_stats:
                        delta_conflicts = stats['conflicts'] - previous_stats.get('conflicts', 0)
                        delta_time = stats['seconds'] - previous_stats.get('seconds', 0)

                        if delta_time > 0:
                            recent_rate = delta_conflicts / delta_time
                            print(f"  최근 {interval}초: {recent_rate:,.0f} conflicts/sec")

                    previous_stats = stats.copy()

                else:
                    print("통계 정보를 가져올 수 없습니다.")

                # 지속 시간 확인
                if duration and (time.time() - start_watch) >= duration:
                    print(f"\n모니터링 종료 (지속 시간: {duration}초)")
                    break

                # 대기
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n모니터링 중단됨.")

    def _check_final_status(self) -> bool:
        """실험이 완료되었는지 확인"""
        if not self.output_file.exists():
            return False

        try:
            with open(self.output_file, 'r') as f:
                content = f.read()

            # SAT/UNSAT 결과 확인
            if 's SATISFIABLE' in content or 's UNSATISFIABLE' in content:
                return True

            return False

        except Exception:
            return False


class ExperimentProgressTracker:
    """
    여러 실험의 진행 상황을 추적
    """

    def __init__(self, intermediate_results_dir: str = "./intermediate_results"):
        """
        Parameters
        ----------
        intermediate_results_dir : str
            중간 결과 디렉토리
        """
        self.intermediate_results_dir = Path(intermediate_results_dir)

    def get_all_running_experiments(self) -> List[Dict]:
        """
        실행 중인 모든 실험 찾기

        Returns
        -------
        List[Dict]
            실험 정보 리스트
        """
        running_experiments = []

        if not self.intermediate_results_dir.exists():
            return running_experiments

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

                output_file = dir_path / f"{dir_name}.cnf.out"

                if not output_file.exists():
                    continue

                # 모니터로 실행 여부 확인
                monitor = KissatMonitor(str(output_file))
                progress = monitor.get_progress_info()

                if progress['is_running']:
                    running_experiments.append({
                        'code_params': (n, k, d_min),
                        'output_file': output_file,
                        'monitor': monitor,
                        'progress': progress
                    })

            except (ValueError, IndexError):
                continue

        return running_experiments

    def print_all_running_status(self) -> None:
        """실행 중인 모든 실험의 상태 출력"""
        experiments = self.get_all_running_experiments()

        if not experiments:
            print("\n실행 중인 실험이 없습니다.")
            return

        print(f"\n{'='*80}")
        print(f" 실행 중인 실험 ({len(experiments)}개)")
        print(f"{'='*80}\n")

        for exp in experiments:
            n, k, d_min = exp['code_params']
            code_str = f"[{n}, {k}, {d_min}]"

            progress = exp['progress']
            stats = progress.get('current_stats')

            print(f"코드: {code_str}")

            if stats:
                elapsed = timedelta(seconds=int(progress['elapsed_time']))
                print(f"  경과 시간: {elapsed}")
                print(f"  메모리: {stats['memory_mb']:.1f} MB")
                print(f"  Conflicts: {stats['conflicts']:,}")
                print(f"  Restarts: {stats['restarts']:,}")

                rate = progress.get('rate')
                if rate:
                    print(f"  속도: {rate['conflicts_per_sec']:,.0f} conflicts/sec")

            print()

    def watch_all(self, interval: int = 30) -> None:
        """
        모든 실행 중인 실험 모니터링

        Parameters
        ----------
        interval : int
            업데이트 간격 (초)
        """
        print(f"\n{'='*80}")
        print(f" 전체 실험 모니터링")
        print(f"{'='*80}")
        print(f"업데이트 간격: {interval}초")
        print(f"Ctrl+C로 종료\n")

        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')

                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"\n업데이트: {current_time}")

                self.print_all_running_status()

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n모니터링 중단됨.")


def main():
    """메인 함수"""
    import sys

    if len(sys.argv) < 2:
        print("""
사용법:
  python experiment_monitor.py <command> [options]

Commands:
  status              - 실행 중인 모든 실험 상태 확인
  watch <output_file> - 특정 실험 모니터링
  watch-all           - 모든 실험 모니터링

Examples:
  python experiment_monitor.py status
  python experiment_monitor.py watch ./intermediate_results/code_35_10_13/code_35_10_13.cnf.out
  python experiment_monitor.py watch-all
        """)
        return

    command = sys.argv[1]

    if command == 'status':
        tracker = ExperimentProgressTracker()
        tracker.print_all_running_status()

    elif command == 'watch' and len(sys.argv) >= 3:
        output_file = sys.argv[2]
        interval = int(sys.argv[3]) if len(sys.argv) >= 4 else 10

        monitor = KissatMonitor(output_file)
        monitor.watch(interval=interval)

    elif command == 'watch-all':
        interval = int(sys.argv[2]) if len(sys.argv) >= 3 else 30

        tracker = ExperimentProgressTracker()
        tracker.watch_all(interval=interval)

    else:
        print(f"알 수 없는 명령어: {command}")


if __name__ == "__main__":
    main()
