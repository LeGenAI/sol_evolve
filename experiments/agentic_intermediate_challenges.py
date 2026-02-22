#!/usr/bin/env python3
"""
에이전틱 중간 도전 문제들: 미해결 문제로 가는 지능형 경로

[16,7,6] 성공 → [22,11,7], [28,14,8] → [32,14,9] → [35,10,13]

이 버전은 다음을 포함합니다:
1. 지능형 SAT 솔버 선택 (AgenticSolverSelector)
2. 다중 SAT 솔버 지원 (kissat, cadical, glucose 등)
3. 실험 진행률 추적 및 최적화 제안
4. 자동 병렬 실행 모드
5. 최적화된 메모리 관리
"""

import sys
import time
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add kissat_code_search to path
sys.path.insert(0, '/Users/baegjaehyeon/CodeEvolve/src/kissat_code_search')

from engines.code_generator import CodeGenerator
from engines.sat_solver_interface import SATSolver, AgenticSolverSelector
from engines.result_parser import ResultParser


class AgenticChallengeManager:
    """
    에이전틱 도전 관리자 - 지능형 문제 선택 및 최적화
    """
    
    def __init__(self):
        self.agentic_selector = AgenticSolverSelector()
        self.sat_solvers = ["kissat", "cadical", "glucose"]  # Default solver list
        self.supported_problems = [
            (16, 7, 6, "쉬움", "< 1분", "✓ 성공"),
            (18, 9, 6, "쉬움-중간", "1-5분", "도전 1"),
            (20, 10, 6, "중간", "5-15분", "도전 2"),
            (22, 11, 7, "중상", "15-30분", "🎯 첫 번째 목표"),
            (24, 12, 7, "중상", "30-60분", "도전 3"),
            (26, 13, 8, "어려움", "1-2시간", "도전 4"),
            (28, 14, 8, "어려움", "2-4시간", "🎯 두 번째 목표"),
            (30, 15, 8, "매우 어려움", "4-8시간", "도전 5"),
            (32, 14, 9, "매우 어려움", "수일?", "🏆 최종 목표"),
            (35, 10, 13, "극도로 어려움", "수일~월?", "🚀 최신 목표"),  # New target!
        ]

    def analyze_problems(self, problems: List[Tuple]):
        """문제 분석 및 비교"""
        print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║      에이전틱 도전 문제 분석: 지능형 문제 해결 전략              ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
        """)

        print("="*100)
        print(" 문제 상세 분석표 (에이전틱 추천 포함)")
        print("="*100)

        print(f"\n{'코드':12s} {'변수':8s} {'메시지':10s} {'Singleton':10s} {'난이도':10s} {'예상시간':12s} {'상태':10s} {'에이전틱 추천':15s}")
        print("-"*100)

        analyses = []
        for n, k, d_min, difficulty, est_time, status in problems:
            singleton = n - k + 1
            vars_count = k * (n - k)
            num_messages = 2**k - 1

            feasible = d_min <= singleton
            feasible_str = "✓" if feasible else "✗"

            code_str = f"[{n},{k},{d_min}]"
            
            # 에이전틱 추천 계산
            estimated_complexity = (vars_count * num_messages) / 1000000  # 백만 단위
            if estimated_complexity < 1:
                recommendation = "고속 해결 가능"
            elif estimated_complexity < 10:
                recommendation = "중간 속도"
            elif estimated_complexity < 100:
                recommendation = "고성능 필요"
            else:
                recommendation = "고도병렬 필요"

            print(f"{code_str:12s} {vars_count:6d}   {num_messages:8,}   "
                  f"{d_min}≤{singleton} {feasible_str:2s}   {difficulty:10s} {est_time:12s} {status:10s} {recommendation:15s}")

            analyses.append({
                'n': n, 'k': k, 'd_min': d_min,
                'vars': vars_count,
                'messages': num_messages,
                'singleton': singleton,
                'feasible': feasible,
                'difficulty': difficulty,
                'est_time': est_time,
                'status': status,
                'complexity': estimated_complexity,
                'recommendation': recommendation
            })

        return analyses

    def select_problems(self, analyses):
        """도전할 문제 선택 (에이전틱 추천 포함)"""
        print("\n" + "="*100)
        print(" 도전할 문제 선택 (에이전틱 추천 기반)")
        print("="*100)

        print("\n🎯 에이전틱 추천 순서:")
        # 추천 순서: 먼저 간단한 문제로 솔버 성능 테스트, 그 다음 새 문제
        recommended = [
            (1, "[18, 9, 6]", "쉬움", "워밍업 및 솔버 벤치마크"),
            (2, "[20, 10, 6]", "중간", "성능 확인"),
            (3, "[22, 11, 7]", "중상", "🎯 첫 번째 목표"),
            (4, "[28, 14, 8]", "어려움", "🎯 검증된 목표"),
            (5, "[35, 10, 13]", "최신 도전", "🚀 최신 목표 - 다중 솔버 필요"),
        ]

        for i, code, diff, note in recommended:
            print(f"  {i}. {code:12s} - {diff:12s} ({note})")

        print("\n🔍 선택 옵션:")
        print("  1: [18, 9, 6] 시작 (가장 안전 - 솔버 테스트용)")
        print("  2: [20, 10, 6] 시작")
        print("  3: [22, 11, 7] 직행 (도전적)")
        print("  4: [28, 14, 8] 직행 (검증된 목표)")
        print("  5: [35, 10, 13] 최신 도전 (고도 병렬/다중 솔버)")
        print("  a: 모두 순차 실행 (장시간 소요)")
        print("  p: 병렬 실행 모드 (다중 CPU 활용)")
        print("  c: 사용자 정의")
        print("  m: 다중 해 탐색 모드 (여러 inequivalent codes 찾기)")

        choice = input("\n선택: ").strip().lower()

        selected = []
        multiple_mode = False
        num_solutions = 1
        parallel_mode = False

        if choice == '1':
            selected = [(18, 9, 6)]
        elif choice == '2':
            selected = [(20, 10, 6)]
        elif choice == '3':
            selected = [(22, 11, 7)]
        elif choice == '4':
            selected = [(28, 14, 8)]
        elif choice == '5':
            selected = [(35, 10, 13)]  # NEW TARGET!
        elif choice == 'a':
            selected = [(18, 9, 6), (20, 10, 6), (22, 11, 7), (28, 14, 8)]
        elif choice == 'p':
            selected = [(18, 9, 6), (20, 10, 6), (22, 11, 7)]
            parallel_mode = True
        elif choice == 'm':
            print("\n다중 해 탐색 모드")
            print("="*80)
            try:
                n = int(input("  n (부호어 길이): "))
                k = int(input("  k (메시지 길이): "))
                d = int(input("  d_min (최소 거리): "))
                num_solutions = int(input("  찾을 해의 개수 (기본값 10): ") or "10")
                selected = [(n, k, d)]
                multiple_mode = True
            except:
                print("잘못된 입력. 기본값 [28, 14, 8], 10개 사용.")
                selected = [(28, 14, 8)]
                num_solutions = 10
                multiple_mode = True
        elif choice == 'c':
            print("\n사용자 정의 입력:")
            try:
                n = int(input("  n (부호어 길이): "))
                k = int(input("  k (메시지 길이): "))
                d = int(input("  d_min (최소 거리): "))
                selected = [(n, k, d)]
            except:
                print("잘못된 입력. 기본값 [18, 9, 6] 사용.")
                selected = [(18, 9, 6)]
        else:
            print(f"잘못된 선택. 기본값 [18, 9, 6] 사용.")
            selected = [(18, 9, 6)]

        return selected, multiple_mode, num_solutions, parallel_mode


class AgenticCodeSolver:
    """
    에이전틱 코드 솔버 - 지능형 솔버 선택 및 최적화
    """
    
    def __init__(self, n: int, k: int, d_min: int):
        self.n = n
        self.k = k
        self.d_min = d_min
        self.agentic_selector = AgenticSolverSelector()
        
    def solve_with_agentic_approach(self, 
                                  solver_paths: Dict[str, str] = None,
                                  timeout: int = 3600,
                                  verbose: bool = True) -> Dict:
        """
        에이전틱 접근으로 코드 해결
        
        Parameters
        ----------
        solver_paths : dict
            solver_type -> path mapping
        timeout : int
            총 제한 시간 (초)
        verbose : bool
            상세 출력 여부
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"에이전틱 접근 시작: [{self.n}, {self.k}, {self.d_min}]")
            print(f"{'='*80}")
        
        # 1단계: CNF 생성
        if verbose:
            print(f"\n[1단계] CNF 인코딩 시작")
        
        start_cnf = time.time()
        generator = CodeGenerator(
            n=self.n,
            k=self.k,
            d_min=self.d_min,
            systematic=True,
            working_dir=f"./agentic_results/code_{self.n}_{self.k}_{self.d_min}"
        )
        
        try:
            cnf_file = generator.generate_cnf()
            cnf_time = time.time() - start_cnf
        except Exception as e:
            return {
                'status': 'CNF_ERROR',
                'error': str(e),
                'time': time.time() - start_cnf
            }
        
        if verbose:
            size_mb = os.path.getsize(cnf_file) / 1024 / 1024 if os.path.exists(cnf_file) else 0
            print(f"  ✓ CNF 생성 완료: {cnf_time:.2f}s, {size_mb:.2f}MB")
            print(f"  ✓ 변수: {generator.encoder.var_counter - 1:,}개")
            print(f"  ✓ 절: {len(generator.encoder.clauses):,}개")
        
        # 2단계: 에이전틱 솔버 추천
        if verbose:
            print(f"\n[2단계] 에이전틱 솔버 추천")
        
        recommendation = self.agentic_selector.recommend_solver(cnf_file, self.n, self.k, self.d_min)
        if verbose:
            print(f"  추천 솔버: {recommendation['solver']}")
            print(f"  추천 이유: {recommendation['reason']}")
        
        # 3단계: 솔버 실행
        if verbose:
            print(f"\n[3단계] SAT 해결 시도")
            print(f"  시작: {time.strftime('%H:%M:%S')}")
        
        start_solve = time.time()
        # Need to create a new AgenticSolverSelector that can handle specific solver paths
        result = self.agentic_selector.solve_with_best_solver(
            cnf_file=cnf_file,
            n=self.n,
            k=self.k,
            d_min=self.d_min,
            timeout=timeout,
            verbose=verbose
        )
        
        solve_time = time.time() - start_solve
        
        # 4단계: 결과 처리
        if verbose:
            print(f"\n[4단계] 결과 처리")
            print(f"  상태: {result['status']}")
            print(f"  해결 시간: {solve_time:.2f}초")
            print(f"  사용 솔버: {result.get('best_solver', result.get('solver_used', 'unknown'))}")
        
        if result['status'] == 'SAT':
            # 결과 파싱 및 검증
            parser = ResultParser(self.n, self.k, self.d_min, systematic=True)
            G = parser.parse_sat_model(result['model'], generator.encoder.var_map)
            
            verification = parser.verify_generator_matrix(G, verbose=False)
            
            if verbose:
                print(f"  ✓ 생성행렬 복원 완료")
                print(f"  ✓ 검증: rank={verification['rank']}/{self.k}, d_min={verification['minimum_distance']}")
            
            result['generator_matrix'] = G
            result['verification'] = verification
            
            # 결과 저장
            result_file = f"./agentic_results/code_{self.n}_{self.k}_{self.d_min}/result_{int(time.time())}.npy"
            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            np.save(result_file, G)
            
            if verbose:
                print(f"  ✓ 결과 저장: {result_file}")
        
        result['cnf_file'] = cnf_file
        result['cnf_time'] = cnf_time
        result['solve_time'] = solve_time
        result['total_time'] = cnf_time + solve_time
        
        return result


def run_agentic_challenge(n: int, k: int, d_min: int, timeout: int = 3600):
    """에이전틱 도전 실행"""
    print(f"\n{'='*80}")
    print(f" [{n}, {k}, {d_min}] 에이전틱 도전")
    print(f"{'='*80}")

    # 분석
    singleton = n - k + 1
    vars_count = k * (n - k)
    num_messages = 2**k - 1

    print(f"\n[1단계] 이론적 분석")
    print(f"  Singleton Bound: {d_min} ≤ {singleton} → {'✓' if d_min <= singleton else '✗'}")
    print(f"  변수 수: {vars_count:,}")
    print(f"  메시지 수: {num_messages:,}")
    print(f"  추정 복잡도: {vars_count * num_messages:,}")
    print(f"  탐색 공간: 2^{vars_count}")

    if d_min > singleton:
        print(f"\n✗ Singleton Bound 위반! 이 코드는 존재하지 않습니다.")
        return None

    # 에이전틱 솔버 실행
    solver = AgenticCodeSolver(n, k, d_min)
    result = solver.solve_with_agentic_approach(timeout=timeout, verbose=True)

    # 결과 출력
    print(f"\n[최종] 결과 분석")
    print(f"  상태: {result['status']}")
    print(f"  총 시간: {result['total_time']:.2f}초")
    print(f"  CNF 시간: {result['cnf_time']:.2f}초")
    print(f"  해결 시간: {result['solve_time']:.2f}초")
    
    if result['status'] == 'SAT':
        print(f"\n" + "🎉"*50)
        print(" "*20 + "에이전틱 해결 성공!")
        print("🎉"*50)

        G = result['generator_matrix']
        verification = result['verification']

        print(f"\n생성행렬 G ({k}×{n}):")
        if n <= 24:  # 작은 경우만 전체 출력
            print(G)
        else:
            print(f"  (크기가 커서 생략, 파일로 저장됨)")

        print(f"\n검증:")
        print(f"  ✓ Rank: {verification['rank']}/{k}")
        print(f"  ✓ 최소 거리: {verification['minimum_distance']}")
        print(f"  ✓ 목표 달성: {verification['meets_requirement']}")

        properties = ResultParser(n, k, d_min, systematic=True).analyze_code_properties(G)
        print(f"\n부호 속성:")
        print(f"  Code rate: {properties['rate']:.4f}")
        print(f"  오류 정정: {properties['error_correction_capability']} 비트")
        print(f"  오류 검출: {properties['error_detection_capability']} 비트")

        print(f"\n무게 분포 (처음 10개):")
        for i, (w, count) in enumerate(list(properties['weight_distribution'].items())[:10]):
            print(f"  w={w}: {count}개")

        result_file = result.get('result_file', 'not_saved')
        print(f"\n저장된 파일: {result_file}")

        return result

    elif result['status'] == 'UNSAT':
        print(f"\n✗ UNSAT: 해가 존재하지 않음")
        print(f"\n이 파라미터로는 코드가 존재하지 않습니다.")
        print(f"(이론적으로는 가능해 보이지만 실제로는 불가능)")

    elif result['status'] == 'TIMEOUT':
        print(f"\n⏱ 시간 초과 ({timeout}초)")
        print(f"\n제안:")
        print(f"  1. 제한 시간 증가: {timeout * 2}초")
        print(f"  2. 고성능 컴퓨팅 자원 사용")
        print(f"  3. 분할 정복 접근 (Column fixing 기법)")

    else:
        print(f"\n❌ 해결 실패: {result.get('error', '알 수 없는 오류')}")
        
    return result


def run_agentic_challenge_parallel(tasks):
    """병렬로 에이전틱 도전 실행"""
    n, k, d_min, timeout = tasks
    return run_agentic_challenge(n, k, d_min, timeout)


def main():
    """메인 함수 - 에이전틱 도전 관리"""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                  ║
║     에이전틱 중간 도전 시스템 - 지능형 SAT 솔버 선택              ║
║                                                                  ║
║     새로운 목표: [35, 10, 13] 코드 발견 (다중 솔버 활용)         ║
║                                                                  ║
╚════════════════════════════════════════════════════════════════════╝
    """)

    # 디렉토리 생성
    os.makedirs('./agentic_results', exist_ok=True)

    # 문제 관리자 생성
    manager = AgenticChallengeManager()

    # 문제 분석
    analyses = manager.analyze_problems(manager.supported_problems)

    # 문제 선택
    selected, multiple_mode, num_solutions, parallel_mode = manager.select_problems(analyses)

    if not selected:
        print("선택된 문제가 없습니다.")
        return

    print(f"\n선택된 문제: {selected}")
    if multiple_mode:
        print(f"모드: 다중 해 탐색 ({num_solutions}개)")
    if parallel_mode:
        print(f"모드: 병렬 실행 ({min(len(selected), mp.cpu_count())} CPU 사용)")

    # 각 문제 도전
    results = []

    if multiple_mode:
        # 다중 해 탐색은 기존 방식 사용
        for i, (n, k, d_min) in enumerate(selected, 1):
            print(f"\n\n{'#'*80}")
            print(f"# 다중 해 탐색: [{n}, {k}, {d_min}]")
            print(f"{'#'*80}")

            # 제한 시간 설정
            if n <= 20:
                timeout = 600  # 10분
            elif n <= 24:
                timeout = 1800  # 30분
            elif n <= 28:
                timeout = 7200  # 2시간
            elif n == 35:  # NEW TARGET
                timeout = 43200  # 12시간
            else:
                timeout = 43200  # 12시간

            # 사용자 확인
            if n >= 24 or n == 35:  # NEW TARGET
                print(f"\n⚠️  이 문제는 각 해마다 최대 {timeout//60}분 걸릴 수 있습니다.")
                response = input("계속하시겠습니까? (y/n): ")
                if response.lower() != 'y':
                    print("건너뜀.")
                    continue

            from engines.code_generator import CodeGenerator
            generator = CodeGenerator(n=n, k=k, d_min=d_min, systematic=True)
            multiple_results = generator.solve_multiple(
                kissat_path='./kissat/build/kissat',  # 이 부분은 나중에 업데이트 예정
                num_solutions=num_solutions,
                timeout_per_instance=timeout,
                verbose=True
            )
            results.append((n, k, d_min, multiple_results, True))

    else:
        if parallel_mode and len(selected) > 1:
            # 병렬 실행
            print(f"\n병렬 실행 모드 시작 (CPU {min(len(selected), mp.cpu_count())}개 사용)")
            
            # 기본 타임아웃 설정
            timeout_per_task = []
            for n, k, d_min in selected:
                if n <= 20:
                    timeout = 600
                elif n <= 24:
                    timeout = 1800
                elif n <= 28:
                    timeout = 7200
                elif n == 35:  # NEW TARGET
                    timeout = 43200
                else:
                    timeout = 43200
                timeout_per_task.append((n, k, d_min, timeout))
            
            with ThreadPoolExecutor(max_workers=min(len(selected), mp.cpu_count())) as executor:
                futures = {executor.submit(run_agentic_challenge_parallel, task): task for task in timeout_per_task}
                
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        result = future.result()
                        results.append((task[0], task[1], task[2], result, False))
                    except Exception as e:
                        print(f"Task {task} failed with error: {e}")
                        results.append((task[0], task[1], task[2], {'status': 'ERROR', 'error': str(e)}, False))
        else:
            # 순차 실행
            for i, (n, k, d_min) in enumerate(selected, 1):
                print(f"\n\n{'#'*80}")
                print(f"# 도전 {i}/{len(selected)}: [{n}, {k}, {d_min}]")
                print(f"{'#'*80}")

                # 제한 시간 설정
                if n <= 20:
                    timeout = 600  # 10분
                elif n <= 24:
                    timeout = 1800  # 30분
                elif n <= 28:
                    timeout = 7200  # 2시간
                elif n == 35:  # NEW TARGET
                    timeout = 43200  # 12시간
                else:
                    timeout = 43200  # 12시간

                # 사용자 확인
                if n >= 24 or n == 35:  # NEW TARGET
                    print(f"\n⚠️  이 문제는 {timeout//60}분 이상 걸릴 수 있습니다.")
                    response = input("계속하시겠습니까? (y/n): ")
                    if response.lower() != 'y':
                        print("건너뜀.")
                        continue

                result = run_agentic_challenge(n, k, d_min, timeout=timeout)
                results.append((n, k, d_min, result, False))

                # 실패 시 중단할지 물어봄
                if result and result['status'] not in ['SAT']:
                    print(f"\n⚠️  [{n}, {k}, {d_min}]에서 해를 찾지 못했습니다.")
                    if i < len(selected):
                        response = input("다음 문제로 계속하시겠습니까? (y/n): ")
                        if response.lower() != 'y':
                            print("탐색 중단.")
                            break

    # 최종 요약
    print(f"\n\n" + "="*100)
    print(" 에이전틱 도전 최종 요약")
    print("="*100)

    print(f"\n{'코드':15s} {'상태':12s} {'총 시간':15s} {'솔버':15s}")
    print("-"*100)

    for n, k, d_min, result, _ in results:
        code_str = f"[{n}, {k}, {d_min}]"
        if isinstance(result, dict):
            status = result.get('status', 'UNKNOWN')
            solve_time = result.get('total_time', 0)
            
            if solve_time < 60:
                time_str = f"{solve_time:.2f}초"
            elif solve_time < 3600:
                time_str = f"{solve_time/60:.1f}분"
            else:
                time_str = f"{solve_time/3600:.1f}시간"
                
            solver_used = result.get('best_solver', result.get('solver_used', 'N/A'))
        else:
            status = "MULTIPLE" if result else "CNF_ERROR"
            time_str = "N/A"
            solver_used = "N/A"

        symbol = "✓" if status == "SAT" else "✗"
        print(f"{symbol} {code_str:13s} {status:12s} {time_str:15s} {solver_used:15s}")

    # 통계
    found = sum(1 for _, _, _, r, _ in results if isinstance(r, dict) and r.get('status') == 'SAT')
    print(f"\n성공: {found}/{len(results)}")

    if found == len(results):
        print(f"\n🎉 모든 도전 성공! 축하합니다!")
        print(f"\n다음 단계:")
        print(f"  - 더 큰 문제 도전")
        print(f"  - 발견한 코드들 분석 및 패턴 연구") 
        print(f"  - [35,10,13] 코드의 수학적 특성 분석")
        print(f"  - 논문 작성 준비")
    else:
        print(f"\n다음 시도:")
        print(f"  1. 실패한 문제의 제한 시간 증가")
        print(f"  2. 다양한 SAT 솔버 실험")
        print(f"  3. 고성능 컴퓨팅 자원 활용")
        print(f"  4. 분할 정복 접근 전략")
        print(f"  5. 새로운 목표: [35,10,13] 코드 도전!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        print("진행 상황은 ./agentic_results/에 저장되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()