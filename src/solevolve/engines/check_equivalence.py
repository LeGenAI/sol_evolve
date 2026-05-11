#!/usr/bin/env python3
"""
Systematic form 코드들의 equivalence 체크

Systematic form [I_k | P]에서 두 코드가 equivalent하려면:
P₂가 P₁의 열 permutation이어야 함
"""

import numpy as np
from pathlib import Path
from itertools import permutations

def extract_parity_part(G, k):
    """
    Systematic form [I_k | P]에서 parity 부분 P 추출

    Parameters
    ----------
    G : np.ndarray (k × n)
        생성행렬
    k : int
        메시지 길이

    Returns
    -------
    P : np.ndarray (k × (n-k))
        Parity check 부분
    """
    return G[:, k:]

def are_columns_permutation(P1, P2):
    """
    P2가 P1의 열 permutation인지 확인

    Parameters
    ----------
    P1, P2 : np.ndarray (k × m)
        비교할 두 행렬

    Returns
    -------
    bool
        P2가 P1의 열 permutation이면 True
    """
    if P1.shape != P2.shape:
        return False

    k, m = P1.shape

    # 작은 행렬만 완전 탐색 (m ≤ 8 정도)
    if m <= 8:
        # 모든 permutation 시도
        for perm in permutations(range(m)):
            P1_perm = P1[:, perm]
            if np.array_equal(P1_perm, P2):
                return True
        return False
    else:
        # 큰 행렬은 heuristic: 각 열이 P1의 어떤 열과 같은지 확인
        # (완벽하지 않지만 빠름)
        P2_cols = [P2[:, j] for j in range(m)]
        P1_cols = [P1[:, j] for j in range(m)]

        # P2의 각 열이 P1의 어떤 열과 일치하는지 확인
        used = set()
        for p2_col in P2_cols:
            found = False
            for i, p1_col in enumerate(P1_cols):
                if i not in used and np.array_equal(p2_col, p1_col):
                    used.add(i)
                    found = True
                    break
            if not found:
                return False

        return len(used) == m

def check_equivalence_matrix(matrices, k):
    """
    여러 생성행렬들 간의 equivalence 확인

    Parameters
    ----------
    matrices : list of np.ndarray
        생성행렬 리스트
    k : int
        메시지 길이

    Returns
    -------
    equiv_matrix : np.ndarray (n × n) bool
        equiv_matrix[i][j] = True if matrices[i] ≡ matrices[j]
    """
    n = len(matrices)
    equiv_matrix = np.zeros((n, n), dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                equiv_matrix[i, j] = True
            else:
                P_i = extract_parity_part(matrices[i], k)
                P_j = extract_parity_part(matrices[j], k)
                equiv_matrix[i, j] = are_columns_permutation(P_i, P_j)

    return equiv_matrix

def main():
    """
    code_21_10_7_multiple 디렉토리의 5개 생성행렬 equivalence 확인
    """
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║           Systematic Form Codes Equivalence Check                ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    # 파일 로드
    result_dir = Path("intermediate_results/code_21_10_7_multiple")

    matrices = []
    for i in range(1, 6):
        filepath = result_dir / f"code_21_10_7_solution{i}_matrix.npy"
        if filepath.exists():
            G = np.load(filepath)
            matrices.append(G)
            print(f"✓ 해 #{i} 로드: {filepath.name}")
        else:
            print(f"✗ 해 #{i} 파일 없음: {filepath}")

    if len(matrices) < 2:
        print("\n충분한 생성행렬이 없습니다.")
        return

    print(f"\n총 {len(matrices)}개의 생성행렬을 로드했습니다.\n")

    # 파라미터
    n = 21
    k = 10

    # Systematic form 확인
    print("="*70)
    print("Systematic Form 확인")
    print("="*70)

    for i, G in enumerate(matrices, 1):
        I_part = G[:, :k]
        is_systematic = np.array_equal(I_part, np.eye(k, dtype=int))
        print(f"해 #{i}: {'✓ Systematic' if is_systematic else '✗ Not systematic'}")

        if not is_systematic:
            print(f"  첫 {k}열:")
            print(I_part)

    # Parity 부분 비교
    print("\n" + "="*70)
    print("Parity Check 부분 (P) 비교")
    print("="*70)

    for i, G in enumerate(matrices, 1):
        P = extract_parity_part(G, k)
        print(f"\n해 #{i}의 P ({k}×{n-k}):")
        print(P)

    # Equivalence 확인
    print("\n" + "="*70)
    print("Equivalence 행렬")
    print("="*70)
    print("(i,j) = True: 해 #i와 해 #j가 equivalent\n")

    equiv_matrix = check_equivalence_matrix(matrices, k)

    print("     ", end="")
    for j in range(len(matrices)):
        print(f"#{j+1:2d} ", end="")
    print()
    print("    " + "-" * (4 * len(matrices)))

    for i in range(len(matrices)):
        print(f"#{i+1:2d} | ", end="")
        for j in range(len(matrices)):
            symbol = " ✓ " if equiv_matrix[i, j] else " . "
            print(symbol, end="")
        print()

    # Equivalence classes 찾기
    print("\n" + "="*70)
    print("Equivalence Classes")
    print("="*70)

    visited = [False] * len(matrices)
    classes = []

    for i in range(len(matrices)):
        if not visited[i]:
            eq_class = [i]
            visited[i] = True

            for j in range(i + 1, len(matrices)):
                if equiv_matrix[i, j]:
                    eq_class.append(j)
                    visited[j] = True

            classes.append(eq_class)

    print(f"\n총 {len(classes)}개의 equivalence class 발견:\n")

    for i, eq_class in enumerate(classes, 1):
        solutions = [f"#{idx+1}" for idx in eq_class]
        print(f"Class {i}: {', '.join(solutions)} (크기: {len(eq_class)})")

    # 결론
    print("\n" + "="*70)
    print("결론")
    print("="*70)

    num_inequivalent = len(classes)

    if num_inequivalent == len(matrices):
        print(f"✓ 모든 {len(matrices)}개의 코드가 서로 inequivalent합니다!")
        print(f"  각각 다른 equivalence class에 속합니다.")
    else:
        print(f"총 {num_inequivalent}개의 inequivalent codes:")
        for i, eq_class in enumerate(classes, 1):
            rep = eq_class[0] + 1  # 대표원소
            print(f"  Class {i}: 해 #{rep} (및 동치류 {len(eq_class)}개)")

        print(f"\n⚠️  일부 코드들이 equivalent합니다.")
        print(f"  Inequivalent codes를 얻으려면 각 class에서 하나씩만 선택하세요.")

    # 추가 분석: Parity 부분의 차이
    print("\n" + "="*70)
    print("Parity 부분 차이 분석")
    print("="*70)

    for i in range(len(matrices)):
        for j in range(i + 1, len(matrices)):
            P_i = extract_parity_part(matrices[i], k)
            P_j = extract_parity_part(matrices[j], k)

            diff_count = np.sum(P_i != P_j)
            total_elements = P_i.size

            equiv_str = "≡" if equiv_matrix[i, j] else "≢"

            print(f"해 #{i+1} vs 해 #{j+1} {equiv_str}: "
                  f"{diff_count}/{total_elements} 다른 원소 "
                  f"({100*diff_count/total_elements:.1f}%)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
