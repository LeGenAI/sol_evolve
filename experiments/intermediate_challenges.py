#!/usr/bin/env python3
"""
중간 도전 문제들: 미해결 문제로 가는 길

[16,7,6] 성공 → [22,11,7], [28,14,8] → [32,14,9]
"""

import sys
sys.path.insert(0, '/Users/baegjaehyeon/CodeEvolve/src/kissat_code_search')

from engines.code_generator import CodeGenerator
import time
import os

KISSAT_PATH = '/Users/baegjaehyeon/CodeEvolve/src/kissat_code_search/kissat/build/kissat'

# 중간 도전 문제들
INTERMEDIATE_PROBLEMS = [
    # (n, k, d_min, 난이도, 예상시간, 상태)
    (16, 7, 6, "쉬움", "< 1분", "✓ 성공"),
    (18, 9, 6, "쉬움-중간", "1-5분", "도전 1"),
    (20, 10, 6, "중간", "5-15분", "도전 2"),
    (22, 11, 7, "중상", "15-30분", "🎯 첫 번째 목표"),
    (24, 12, 7, "중상", "30-60분", "도전 3"),
    (26, 13, 8, "어려움", "1-2시간", "도전 4"),
    (28, 14, 8, "어려움", "2-4시간", "🎯 두 번째 목표"),
    (30, 15, 8, "매우 어려움", "4-8시간", "도전 5"),
    (32, 14, 9, "매우 어려움", "수일?", "🏆 최종 목표"),
]


def analyze_and_compare(problems):
    """문제들 분석 및 비교"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║        중간 도전 문제들: [16,7,6] → [22,11,7] → [32,14,9]       ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    print("="*80)
    print(" 문제 분석표")
    print("="*80)

    # 헤더
    print(f"\n{'코드':12s} {'변수':8s} {'메시지':10s} {'Singleton':10s} {'난이도':12s} {'예상시간':12s} {'상태':10s}")
    print("-"*80)

    analyses = []
    for n, k, d_min, difficulty, est_time, status in problems:
        singleton = n - k + 1
        vars_count = k * (n - k)
        num_messages = 2**k - 1

        feasible = d_min <= singleton
        feasible_str = "✓" if feasible else "✗"

        code_str = f"[{n},{k},{d_min}]"

        print(f"{code_str:12s} {vars_count:6d}   {num_messages:8,}   "
              f"{d_min}≤{singleton} {feasible_str:2s}   {difficulty:12s} {est_time:12s} {status:10s}")

        analyses.append({
            'n': n, 'k': k, 'd_min': d_min,
            'vars': vars_count,
            'messages': num_messages,
            'singleton': singleton,
            'feasible': feasible,
            'difficulty': difficulty,
            'est_time': est_time,
            'status': status
        })

    return analyses


def select_problems(analyses):
    """도전할 문제 선택"""
    print("\n" + "="*80)
    print(" 도전할 문제 선택")
    print("="*80)

    print("\n추천 순서:")
    recommended = [
        (1, "[18, 9, 6]", "쉬움", "워밍업"),
        (2, "[20, 10, 6]", "중간", "다음 단계"),
        (3, "[22, 11, 7]", "중상", "🎯 첫 번째 목표"),
        (4, "[24, 12, 7]", "중상", "중간 점검"),
        (5, "[28, 14, 8]", "어려움", "🎯 두 번째 목표"),
    ]

    for i, code, diff, note in recommended:
        print(f"  {i}. {code:12s} - {diff:12s} ({note})")

    print("\n선택 옵션:")
    print("  1: [18, 9, 6] 시작 (가장 안전)")
    print("  2: [20, 10, 6] 시작")
    print("  3: [22, 11, 7] 직행 (도전적)")
    print("  4: [28, 14, 8] 직행 (매우 도전적)")
    print("  a: 모두 순차 실행 (장시간 소요)")
    print("  c: 사용자 정의")
    print("  m: 다중 해 탐색 모드 (여러 inequivalent codes 찾기)")

    choice = input("\n선택: ").strip().lower()

    selected = []
    multiple_mode = False
    num_solutions = 1

    if choice == '1':
        selected = [(18, 9, 6)]
    elif choice == '2':
        selected = [(20, 10, 6)]
    elif choice == '3':
        selected = [(22, 11, 7)]
    elif choice == '4':
        selected = [(28, 14, 8)]
    elif choice == 'a':
        selected = [(18, 9, 6), (20, 10, 6), (22, 11, 7), (24, 12, 7), (28, 14, 8)]
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

    return selected, multiple_mode, num_solutions


def run_challenge_multiple(n, k, d_min, num_solutions=10, timeout=3600):
    """다중 해 탐색"""
    print("\n" + "="*80)
    print(f" [{n}, {k}, {d_min}] 다중 해 탐색 도전")
    print("="*80)

    # 분석
    singleton = n - k + 1
    vars_count = k * (n - k)
    num_messages = 2**k - 1

    print(f"\n[1단계] 이론적 분석")
    print(f"  Singleton Bound: {d_min} ≤ {singleton} → {'✓' if d_min <= singleton else '✗'}")
    print(f"  변수 수: {vars_count:,}")
    print(f"  메시지 수: {num_messages:,}")
    print(f"  탐색 공간: 2^{vars_count}")
    print(f"  목표: {num_solutions}개의 서로 다른 생성행렬")

    if d_min > singleton:
        print(f"\n✗ Singleton Bound 위반! 이 코드는 존재하지 않습니다.")
        return None

    # Generator 생성
    print(f"\n[2단계] Code Generator 초기화")
    generator = CodeGenerator(
        n=n,
        k=k,
        d_min=d_min,
        systematic=True,
        working_dir=f"./intermediate_results/code_{n}_{k}_{d_min}_multiple"
    )

    # Singleton bound 확인
    generator.compare_with_singleton_bound()

    # 다중 해 탐색
    print(f"\n[3단계] 다중 해 탐색 시작")
    print(f"시작 시각: {time.strftime('%H:%M:%S')}")

    results = generator.solve_multiple(
        kissat_path=KISSAT_PATH,
        num_solutions=num_solutions,
        timeout_per_instance=timeout,
        verbose=True
    )

    return results


def run_challenge(n, k, d_min, timeout=3600):
    """도전 실행"""
    print("\n" + "="*80)
    print(f" [{n}, {k}, {d_min}] 코드 탐색 도전")
    print("="*80)

    # 분석
    singleton = n - k + 1
    vars_count = k * (n - k)
    num_messages = 2**k - 1

    print(f"\n[1단계] 이론적 분석")
    print(f"  Singleton Bound: {d_min} ≤ {singleton} → {'✓' if d_min <= singleton else '✗'}")
    print(f"  변수 수: {vars_count:,}")
    print(f"  메시지 수: {num_messages:,}")
    print(f"  탐색 공간: 2^{vars_count}")

    if d_min > singleton:
        print(f"\n✗ Singleton Bound 위반! 이 코드는 존재하지 않습니다.")
        return None

    # Generator 생성
    print(f"\n[2단계] Code Generator 초기화")
    generator = CodeGenerator(
        n=n,
        k=k,
        d_min=d_min,
        systematic=True,
        working_dir=f"./intermediate_results/code_{n}_{k}_{d_min}"
    )

    # Singleton bound 확인
    generator.compare_with_singleton_bound()

    # CNF 생성
    print(f"\n[3단계] CNF 인코딩")
    print(f"시작 시각: {time.strftime('%H:%M:%S')}")

    start_cnf = time.time()
    try:
        cnf_file = generator.generate_cnf()
        cnf_time = time.time() - start_cnf

        if os.path.exists(cnf_file):
            size_mb = os.path.getsize(cnf_file) / 1024 / 1024
            print(f"\n✓ CNF 생성 완료")
            print(f"  시간: {cnf_time:.2f}초")
            print(f"  파일: {cnf_file}")
            print(f"  크기: {size_mb:.2f} MB")
            print(f"  변수: {generator.encoder.var_counter - 1:,}개")
            print(f"  절: {len(generator.encoder.clauses):,}개")
    except Exception as e:
        print(f"\n✗ CNF 생성 실패: {e}")
        return None

    # Kissat 실행
    print(f"\n[4단계] Kissat SAT 솔버 실행")
    print(f"  제한 시간: {timeout}초 ({timeout//60}분)")
    print(f"  시작: {time.strftime('%H:%M:%S')}")

    if timeout > 600:
        print(f"\n⚠️  이 문제는 시간이 오래 걸릴 수 있습니다.")
        print(f"   중단하려면 Ctrl+C를 누르세요.")

    start_solve = time.time()
    result = generator.solve(
        kissat_path=KISSAT_PATH,
        cnf_file=cnf_file,
        timeout=timeout,
        verbose=True
    )
    solve_time = time.time() - start_solve

    # 결과 출력
    print(f"\n[5단계] 결과 분석")
    print(f"  종료: {time.strftime('%H:%M:%S')}")
    print(f"  Kissat 시간: {solve_time:.2f}초 ({solve_time/60:.2f}분)")
    print(f"  총 시간: {cnf_time + solve_time:.2f}초")
    print(f"  상태: {result['status']}")

    if result['status'] == 'SAT':
        print(f"\n" + "🎉"*40)
        print(" "*15 + "해 발견!!!")
        print("🎉"*40)

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

        properties = generator.parser.analyze_code_properties(G)
        print(f"\n부호 속성:")
        print(f"  Code rate: {properties['rate']:.4f}")
        print(f"  오류 정정: {properties['error_correction_capability']} 비트")
        print(f"  오류 검출: {properties['error_detection_capability']} 비트")

        print(f"\n무게 분포 (처음 10개):")
        for i, (w, count) in enumerate(list(properties['weight_distribution'].items())[:10]):
            print(f"  w={w}: {count}개")

        print(f"\n저장된 파일:")
        print(f"  생성행렬: intermediate_results/code_{n}_{k}_{d_min}/code_{n}_{k}_{d_min}_matrix.npy")
        print(f"  상세 결과: intermediate_results/code_{n}_{k}_{d_min}/code_{n}_{k}_{d_min}_result.txt")

        return result

    elif result['status'] == 'UNSAT':
        print(f"\n✗ UNSAT: 해가 존재하지 않음")
        print(f"\n이 파라미터로는 코드가 존재하지 않습니다.")
        print(f"(이론적으로는 가능해 보이지만 실제로는 불가능)")

    elif result['status'] == 'TIMEOUT':
        print(f"\n⏱ 시간 초과 ({timeout}초)")
        print(f"\n제안:")
        print(f"  1. 제한 시간 증가: timeout={timeout * 2}초 ({timeout * 2 // 60}분)")
        print(f"  2. 더 강력한 컴퓨터 사용")
        print(f"  3. 추가 최적화 구현 (Cardinality Networks, 대칭성 파괴)")

    return result


def main():
    """메인 함수"""
    # 디렉토리 생성
    os.makedirs('./intermediate_results', exist_ok=True)

    # 문제 분석
    analyses = analyze_and_compare(INTERMEDIATE_PROBLEMS)

    # 문제 선택
    selected, multiple_mode, num_solutions = select_problems(analyses)

    if not selected:
        print("선택된 문제가 없습니다.")
        return

    print(f"\n선택된 문제: {selected}")
    if multiple_mode:
        print(f"모드: 다중 해 탐색 ({num_solutions}개)")

    # 각 문제 도전
    results = []

    if multiple_mode:
        # 다중 해 탐색 모드
        for i, (n, k, d_min) in enumerate(selected, 1):
            print(f"\n\n{'#'*80}")
            print(f"# 다중 해 탐색: [{n}, {k}, {d_min}]")
            print(f"{'#'*80}")

            # 제한 시간 설정 (문제 크기에 따라)
            if n <= 20:
                timeout = 600  # 10분
            elif n <= 24:
                timeout = 1800  # 30분
            elif n <= 28:
                timeout = 7200  # 2시간
            else:
                timeout = 43200  # 12시간

            # 사용자 확인
            if n >= 24:
                print(f"\n⚠️  이 문제는 각 해마다 최대 {timeout//60}분 걸릴 수 있습니다.")
                print(f"    총 예상 시간: 최대 {(timeout * num_solutions)//60}분")
                response = input("계속하시겠습니까? (y/n): ")
                if response.lower() != 'y':
                    print("건너뜀.")
                    continue

            multiple_results = run_challenge_multiple(n, k, d_min, num_solutions=num_solutions, timeout=timeout)
            results.append((n, k, d_min, multiple_results, True))  # True = multiple mode

        # 다중 해 요약
        print(f"\n\n" + "="*80)
        print(" 다중 해 탐색 최종 요약")
        print("="*80)

        for n, k, d_min, multi_results, _ in results:
            code_str = f"[{n}, {k}, {d_min}]"
            print(f"\n{code_str}:")
            print(f"  총 {len(multi_results)}개의 서로 다른 생성행렬 발견")

            avg_time = sum(r['solver_time'] for r in multi_results) / len(multi_results) if multi_results else 0
            total_time = sum(r['solver_time'] for r in multi_results)

            print(f"  평균 탐색 시간: {avg_time:.2f}초")
            print(f"  총 탐색 시간: {total_time:.2f}초 ({total_time/60:.1f}분)")

            print(f"\n  저장된 파일:")
            for i, res in enumerate(multi_results, 1):
                print(f"    해 #{i}: code_{n}_{k}_{d_min}_solution{i}_matrix.npy")

    else:
        # 단일 해 탐색 모드 (기존)
        for i, (n, k, d_min) in enumerate(selected, 1):
            print(f"\n\n{'#'*80}")
            print(f"# 도전 {i}/{len(selected)}: [{n}, {k}, {d_min}]")
            print(f"{'#'*80}")

            # 제한 시간 설정 (문제 크기에 따라)
            if n <= 20:
                timeout = 600  # 10분
            elif n <= 24:
                timeout = 1800  # 30분
            elif n <= 28:
                timeout = 7200  # 2시간
            else:
                timeout = 43200  # 12시간

            # 사용자 확인
            if n >= 24:
                print(f"\n⚠️  이 문제는 {timeout//60}분 이상 걸릴 수 있습니다.")
                response = input("계속하시겠습니까? (y/n): ")
                if response.lower() != 'y':
                    print("건너뜀.")
                    continue

            result = run_challenge(n, k, d_min, timeout=timeout)
            results.append((n, k, d_min, result, False))  # False = single mode

            # 실패 시 중단할지 물어봄
            if result and result['status'] not in ['SAT']:
                print(f"\n⚠️  [{n}, {k}, {d_min}]에서 해를 찾지 못했습니다.")
                if i < len(selected):
                    response = input("다음 문제로 계속하시겠습니까? (y/n): ")
                    if response.lower() != 'y':
                        print("탐색 중단.")
                        break

        # 단일 해 최종 요약
        print(f"\n\n" + "="*80)
        print(" 최종 요약")
        print("="*80)

        print(f"\n{'코드':15s} {'상태':12s} {'시간':15s}")
        print("-"*80)

        for n, k, d_min, result, _ in results:
            code_str = f"[{n}, {k}, {d_min}]"
            if result:
                status = result['status']
                solve_time = result.get('solver_time', 0)

                if solve_time < 60:
                    time_str = f"{solve_time:.2f}초"
                elif solve_time < 3600:
                    time_str = f"{solve_time/60:.1f}분"
                else:
                    time_str = f"{solve_time/3600:.1f}시간"
            else:
                status = "CNF_ERROR"
                time_str = "N/A"

            symbol = "✓" if status == "SAT" else "✗"
            print(f"{symbol} {code_str:13s} {status:12s} {time_str:15s}")

        # 통계
        found = sum(1 for _, _, _, r, _ in results if r and r['status'] == 'SAT')
        print(f"\n성공: {found}/{len(results)}")

        if found == len(results):
            print(f"\n🎉 모든 도전 성공! 축하합니다!")
            print(f"\n다음 단계:")
            print(f"  - 더 큰 문제 도전")
            print(f"  - 발견한 코드들 분석 및 패턴 연구")
            print(f"  - 논문 작성 준비")
        else:
            print(f"\n다음 시도:")
            print(f"  1. 실패한 문제의 제한 시간 증가")
            print(f"  2. 추가 최적화 구현")
            print(f"  3. 다른 접근법 시도")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n사용자에 의해 중단되었습니다.")
        print("진행 상황은 ./intermediate_results/에 저장되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
