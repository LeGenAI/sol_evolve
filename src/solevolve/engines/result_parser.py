"""
SAT Result Parser and Verification

이 모듈은 Kissat 솔버의 결과를 파싱하고 생성행렬을 복원 및 검증합니다.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from itertools import product


class ResultParser:
    """
    SAT 솔버 결과를 생성행렬로 변환하고 검증하는 클래스.
    """

    def __init__(self, n: int, k: int, d_min: int, systematic: bool = True):
        """
        Parameters
        ----------
        n : int
            부호어 길이
        k : int
            메시지 길이
        d_min : int
            목표 최소 거리
        systematic : bool
            체계적 형태 사용 여부
        """
        self.n = n
        self.k = k
        self.d_min = d_min
        self.systematic = systematic

    def parse_sat_model(self, model: Dict[int, bool], var_map: Dict) -> np.ndarray:
        """
        SAT 모델(변수 할당)을 생성행렬 G로 변환.

        Parameters
        ----------
        model : Dict[int, bool]
            SAT 솔버가 반환한 변수 할당
            {var_id: True/False}
        var_map : Dict
            CNFEncoder의 변수 매핑 정보
            {('P', i, j): var_id} 또는 {('G', i, j): var_id}

        Returns
        -------
        np.ndarray
            k × n 생성행렬 G
        """
        G = np.zeros((self.k, self.n), dtype=int)

        if self.systematic:
            # 왼쪽 k×k: 항등행렬
            for i in range(self.k):
                G[i, i] = 1

            # 오른쪽 k×(n-k): 패리티 행렬 P
            reverse_map = {v: k for k, v in var_map.items()}
            for var_id, value in model.items():
                if var_id in reverse_map:
                    key = reverse_map[var_id]
                    if key[0] == 'P':
                        _, i, j = key
                        G[i, self.k + j] = 1 if value else 0
        else:
            # 일반 형태: 전체 행렬
            reverse_map = {v: k for k, v in var_map.items()}
            for var_id, value in model.items():
                if var_id in reverse_map:
                    key = reverse_map[var_id]
                    if key[0] == 'G':
                        _, i, j = key
                        G[i, j] = 1 if value else 0

        return G

    def verify_generator_matrix(self, G: np.ndarray, verbose: bool = True) -> Dict:
        """
        생성행렬 G의 유효성 검증.

        검증 항목:
        1. 행렬 크기 확인
        2. Rank 검증 (선형 독립성)
        3. 최소 거리 계산
        4. 목표 최소 거리와 비교

        Parameters
        ----------
        G : np.ndarray
            k × n 생성행렬
        verbose : bool
            상세 출력 여부

        Returns
        -------
        dict
            {
                'valid': bool,
                'rank': int,
                'minimum_distance': int,
                'meets_requirement': bool,
                'codewords_count': int,
                'messages': list of any issues
            }
        """
        issues = []
        result = {
            'valid': True,
            'rank': 0,
            'minimum_distance': 0,
            'meets_requirement': False,
            'codewords_count': 0,
            'messages': []
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f"생성행렬 검증")
            print(f"{'='*60}\n")
            print(f"목표: [{self.n}, {self.k}, {self.d_min}] 코드")
            print(f"\n생성행렬 G ({self.k}×{self.n}):")
            print(G)
            print()

        # 1. 크기 확인
        if G.shape != (self.k, self.n):
            issues.append(f"잘못된 행렬 크기: {G.shape}, 예상: ({self.k}, {self.n})")
            result['valid'] = False

        # 2. Rank 계산 (GF(2)에서)
        rank = self._compute_rank_gf2(G)
        result['rank'] = rank

        if verbose:
            print(f"Rank: {rank}/{self.k}")

        if rank < self.k:
            issues.append(f"Rank 부족: {rank} < {self.k} (행들이 선형 종속)")
            result['valid'] = False

        # 3. 모든 부호어 생성 및 최소 거리 계산
        codewords = self._generate_all_codewords(G)
        result['codewords_count'] = len(codewords)

        if verbose:
            print(f"부호어 개수: {len(codewords)}")

        if len(codewords) != 2**self.k:
            issues.append(
                f"부호어 개수 불일치: {len(codewords)} != 2^{self.k} = {2**self.k}"
            )
            result['valid'] = False

        # 최소 거리 계산
        min_dist = self._compute_minimum_distance(codewords)
        result['minimum_distance'] = min_dist

        if verbose:
            print(f"최소 거리: {min_dist}")

        # 4. 목표와 비교
        result['meets_requirement'] = (min_dist >= self.d_min)

        if verbose:
            print(f"\n목표 달성: {result['meets_requirement']}")
            if min_dist >= self.d_min:
                print(f"  ✓ 최소 거리 {min_dist} >= {self.d_min}")
            else:
                print(f"  ✗ 최소 거리 {min_dist} < {self.d_min}")
                issues.append(f"최소 거리 부족: {min_dist} < {self.d_min}")

        result['messages'] = issues

        if issues:
            result['valid'] = False
            if verbose:
                print(f"\n문제점:")
                for msg in issues:
                    print(f"  - {msg}")

        if verbose:
            print(f"\n{'='*60}")
            print(f"검증 결과: {'성공 ✓' if result['valid'] and result['meets_requirement'] else '실패 ✗'}")
            print(f"{'='*60}\n")

        return result

    def _compute_rank_gf2(self, G: np.ndarray) -> int:
        """
        GF(2)에서 행렬의 Rank 계산 (가우스 소거법).

        Parameters
        ----------
        G : np.ndarray
            이진 행렬

        Returns
        -------
        int
            Rank
        """
        G_copy = G.copy()
        rows, cols = G_copy.shape
        rank = 0
        pivot_col = 0

        for row in range(rows):
            if pivot_col >= cols:
                break

            # Pivot 찾기
            pivot_row = None
            for r in range(row, rows):
                if G_copy[r, pivot_col] == 1:
                    pivot_row = r
                    break

            if pivot_row is None:
                # 이 열에는 pivot 없음, 다음 열로
                pivot_col += 1
                continue

            # 행 교환
            if pivot_row != row:
                G_copy[[row, pivot_row]] = G_copy[[pivot_row, row]]

            # 소거 (GF(2)에서 덧셈은 XOR)
            for r in range(rows):
                if r != row and G_copy[r, pivot_col] == 1:
                    G_copy[r] = (G_copy[r] + G_copy[row]) % 2

            rank += 1
            pivot_col += 1

        return rank

    def _generate_all_codewords(self, G: np.ndarray) -> np.ndarray:
        """
        생성행렬 G로부터 모든 부호어 생성.

        Parameters
        ----------
        G : np.ndarray
            k × n 생성행렬

        Returns
        -------
        np.ndarray
            (2^k) × n 부호어 행렬
        """
        k = G.shape[0]
        num_codewords = 2**k
        codewords = []

        for msg_int in range(num_codewords):
            # 메시지 벡터 생성
            message = np.array(
                [int(b) for b in format(msg_int, f'0{k}b')],
                dtype=int
            )
            # 부호어 = message × G (GF(2))
            codeword = np.dot(message, G) % 2
            codewords.append(codeword)

        return np.array(codewords, dtype=int)

    def _compute_minimum_distance(self, codewords: np.ndarray) -> int:
        """
        부호어들 간의 최소 해밍 거리 계산.

        선형 부호에서는 0이 아닌 부호어들의 최소 해밍 무게와 같음.

        Parameters
        ----------
        codewords : np.ndarray
            부호어 행렬

        Returns
        -------
        int
            최소 거리
        """
        # 0 벡터를 제외한 모든 부호어의 해밍 무게 계산
        min_weight = float('inf')

        for codeword in codewords:
            weight = np.sum(codeword)
            if weight > 0:  # 0 벡터 제외
                min_weight = min(min_weight, weight)

        return int(min_weight) if min_weight != float('inf') else 0

    def analyze_code_properties(self, G: np.ndarray) -> Dict:
        """
        생성행렬로부터 부호의 다양한 속성 분석.

        Parameters
        ----------
        G : np.ndarray
            생성행렬

        Returns
        -------
        dict
            부호의 다양한 속성들
        """
        properties = {}

        # 기본 파라미터
        properties['n'] = self.n
        properties['k'] = self.k
        properties['rate'] = self.k / self.n

        # 생성행렬 정보
        properties['systematic'] = self.systematic
        if self.systematic:
            parity_matrix = G[:, self.k:]
            properties['parity_matrix'] = parity_matrix
            properties['parity_check_matrix'] = self._compute_parity_check_matrix(G)

        # 부호어 생성 및 분석
        codewords = self._generate_all_codewords(G)
        properties['codewords'] = codewords

        # 무게 분포 (Weight Distribution)
        weight_dist = self._compute_weight_distribution(codewords)
        properties['weight_distribution'] = weight_dist

        # 최소 거리
        properties['minimum_distance'] = min(
            [w for w, count in weight_dist.items() if w > 0],
            default=0
        )

        # 오류 정정 능력
        properties['error_correction_capability'] = (properties['minimum_distance'] - 1) // 2

        # 오류 검출 능력
        properties['error_detection_capability'] = properties['minimum_distance'] - 1

        return properties

    def _compute_parity_check_matrix(self, G: np.ndarray) -> np.ndarray:
        """
        체계적 형태의 생성행렬로부터 패리티 검사 행렬 H 계산.

        G = [I_k | P] 일 때, H = [-P^T | I_{n-k}] = [P^T | I_{n-k}] (GF(2)에서 -1 = 1)

        Parameters
        ----------
        G : np.ndarray
            체계적 형태의 생성행렬

        Returns
        -------
        np.ndarray
            (n-k) × n 패리티 검사 행렬
        """
        if not self.systematic:
            raise ValueError("패리티 검사 행렬은 체계적 형태에서만 계산 가능")

        P = G[:, self.k:]  # k × (n-k)
        P_T = P.T  # (n-k) × k

        # H = [P^T | I_{n-k}]
        I_n_minus_k = np.eye(self.n - self.k, dtype=int)
        H = np.hstack([P_T, I_n_minus_k])

        return H

    def _compute_weight_distribution(self, codewords: np.ndarray) -> Dict[int, int]:
        """
        부호어들의 무게 분포 계산.

        Parameters
        ----------
        codewords : np.ndarray
            부호어 행렬

        Returns
        -------
        dict
            {weight: count} 형태의 무게 분포
        """
        weight_dist = {}

        for codeword in codewords:
            weight = int(np.sum(codeword))
            weight_dist[weight] = weight_dist.get(weight, 0) + 1

        return dict(sorted(weight_dist.items()))

    def export_results(self, G: np.ndarray, filename: str, include_codewords: bool = False):
        """
        검증 결과를 파일로 저장.

        Parameters
        ----------
        G : np.ndarray
            생성행렬
        filename : str
            저장할 파일 경로
        include_codewords : bool
            모든 부호어 포함 여부
        """
        verification = self.verify_generator_matrix(G, verbose=False)
        properties = self.analyze_code_properties(G)

        with open(filename, 'w') as f:
            f.write(f"Binary Linear Code [{self.n}, {self.k}, {self.d_min}]\n")
            f.write("="*60 + "\n\n")

            f.write("생성행렬 G:\n")
            f.write(str(G) + "\n\n")

            f.write("검증 결과:\n")
            f.write(f"  유효성: {verification['valid']}\n")
            f.write(f"  Rank: {verification['rank']}\n")
            f.write(f"  최소 거리: {verification['minimum_distance']}\n")
            f.write(f"  목표 달성: {verification['meets_requirement']}\n\n")

            f.write("부호 속성:\n")
            f.write(f"  Code rate: {properties['rate']:.4f}\n")
            f.write(f"  오류 정정 능력: {properties['error_correction_capability']} 비트\n")
            f.write(f"  오류 검출 능력: {properties['error_detection_capability']} 비트\n\n")

            f.write("무게 분포:\n")
            for weight, count in properties['weight_distribution'].items():
                f.write(f"  w={weight}: {count}개\n")

            if include_codewords:
                f.write(f"\n모든 부호어 ({len(properties['codewords'])}개):\n")
                for i, cw in enumerate(properties['codewords']):
                    cw_str = ''.join(map(str, cw))
                    f.write(f"  {i:03d}: {cw_str}\n")

        print(f"결과 저장 완료: {filename}")


if __name__ == "__main__":
    # 예제: 알려진 [7, 4, 3] 해밍 코드로 테스트
    print("예제: [7, 4, 3] 해밍 코드 검증\n")

    # 표준 해밍 코드 생성행렬
    G_hamming = np.array([
        [1, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1]
    ], dtype=int)

    parser = ResultParser(n=7, k=4, d_min=3, systematic=True)
    verification = parser.verify_generator_matrix(G_hamming, verbose=True)

    # 상세 속성 분석
    properties = parser.analyze_code_properties(G_hamming)
    print("\n추가 속성:")
    print(f"  Code rate: {properties['rate']:.4f}")
    print(f"  오류 정정: 최대 {properties['error_correction_capability']}비트")
    print(f"  오류 검출: 최대 {properties['error_detection_capability']}비트")
    print(f"\n무게 분포: {properties['weight_distribution']}")

    # 결과 저장
    parser.export_results(G_hamming, "hamming_7_4_3_verification.txt", include_codewords=True)
