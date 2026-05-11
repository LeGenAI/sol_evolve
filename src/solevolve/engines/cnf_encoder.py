"""
CNF Encoder for Binary Linear Code Search

이 모듈은 Binary [n, k, d] 코드 탐색 문제를 SAT 문제(CNF)로 인코딩합니다.
"""

import numpy as np
from typing import List, Tuple, Set
from itertools import combinations


class CNFEncoder:
    """
    Binary Linear Code 제약 조건을 DIMACS CNF 형식으로 인코딩하는 클래스.
    """

    def __init__(self, n: int, k: int, d_min: int, systematic: bool = True):
        """
        Parameters
        ----------
        n : int
            부호어 길이 (codeword length)
        k : int
            메시지 길이 (dimension)
        d_min : int
            최소 거리 (minimum distance)
        systematic : bool
            체계적 형태(systematic form) G = [I_k | P] 사용 여부
        """
        self.n = n
        self.k = k
        self.d_min = d_min
        self.systematic = systematic

        # 변수 관리
        self.var_counter = 1  # DIMACS는 1부터 시작
        self.var_map = {}  # (type, index) -> var_id
        self.clauses = []  # CNF 절 리스트

        # 생성행렬 변수 생성
        self._create_generator_variables()

    def _create_generator_variables(self):
        """생성행렬 G의 각 원소에 대한 Boolean 변수 생성."""
        if self.systematic:
            # Systematic form: G = [I_k | P]
            # I_k 부분은 고정되어 있으므로 변수 불필요
            # P 부분만 k × (n-k) 변수 필요
            self.parity_bits = self.n - self.k
            for i in range(self.k):
                for j in range(self.parity_bits):
                    var_id = self.var_counter
                    self.var_map[('P', i, j)] = var_id
                    self.var_counter += 1
        else:
            # 일반 형태: 전체 k × n 변수 필요
            for i in range(self.k):
                for j in range(self.n):
                    var_id = self.var_counter
                    self.var_map[('G', i, j)] = var_id
                    self.var_counter += 1

    def get_generator_variable(self, row: int, col: int) -> int:
        """
        생성행렬 G[row, col]에 해당하는 SAT 변수 ID 반환.

        Systematic form인 경우:
        - col < k: 항등행렬 부분 (상수)
        - col >= k: 패리티 행렬 부분 (변수)
        """
        if self.systematic:
            if col < self.k:
                # 항등행렬 부분: G[i,i] = 1, 나머지 = 0
                # CNF에서 상수는 변수가 아니라 절로 표현
                return None  # 상수는 변수 ID 없음
            else:
                # 패리티 행렬 P의 col - k번째 열
                return self.var_map[('P', row, col - self.k)]
        else:
            return self.var_map[('G', row, col)]

    def encode_minimum_distance_constraints(self):
        """
        최소 거리 제약 조건 인코딩.

        모든 0이 아닌 메시지 m (2^k - 1개)에 대해,
        부호어 c = mG의 해밍 무게가 d_min 이상이어야 함.
        """
        print(f"인코딩 중: 최소 거리 제약 (메시지 수: {2**self.k - 1})")

        # 모든 비영 메시지 벡터 생성
        for msg_int in range(1, 2**self.k):
            message = self._int_to_binary_vector(msg_int, self.k)
            self._encode_single_message_constraint(message)

    def _int_to_binary_vector(self, value: int, length: int) -> np.ndarray:
        """정수를 이진 벡터로 변환."""
        binary_str = format(value, f'0{length}b')
        return np.array([int(b) for b in binary_str], dtype=int)

    def _encode_single_message_constraint(self, message: np.ndarray):
        """
        특정 메시지 m에 대한 제약 조건 인코딩.

        c = mG의 각 비트 c_j를 계산하고,
        w(c) >= d_min 제약을 CNF로 변환.
        """
        # 1단계: 부호어의 각 비트 c_j 계산용 변수 생성
        codeword_vars = []
        for j in range(self.n):
            # c_j = XOR_{i where m_i=1} G[i,j]
            c_var = self._encode_codeword_bit(message, j)
            codeword_vars.append(c_var)

        # 2단계: At-Least-K 제약: codeword_vars 중 d_min개 이상이 True
        self._encode_at_least_k(codeword_vars, self.d_min)

    def _encode_codeword_bit(self, message: np.ndarray, bit_position: int) -> int:
        """
        부호어의 특정 비트 c_j = XOR_{i where m_i=1} G[i,j] 를 변수로 인코딩.

        Returns
        -------
        int
            c_j를 나타내는 SAT 변수 ID
        """
        # 메시지에서 1인 비트의 인덱스 찾기
        active_rows = np.where(message == 1)[0]

        if len(active_rows) == 0:
            # 메시지가 0 (발생하지 않아야 함)
            raise ValueError("메시지는 0이 아니어야 합니다")

        # XOR할 G의 변수들 수집
        g_vars = []
        for i in active_rows:
            if self.systematic and bit_position < self.k:
                # 항등행렬 부분: G[i, bit_position] = (1 if i == bit_position else 0)
                if i == bit_position:
                    g_vars.append(('const', 1))
                # else: 0이므로 XOR에 영향 없음, 추가 안 함
            else:
                var_id = self.get_generator_variable(i, bit_position)
                g_vars.append(('var', var_id))

        # XOR 인코딩
        if len(g_vars) == 1 and g_vars[0][0] == 'const':
            # 상수 1만 있는 경우: 새 변수 생성하고 그것을 True로 고정
            result_var = self.var_counter
            self.var_counter += 1
            self.clauses.append([result_var])  # result_var = True
            return result_var

        # 일반적인 XOR 인코딩 (Tseitin 변환 사용)
        return self._encode_xor(g_vars)

    def _encode_xor(self, variables: List[Tuple[str, int]]) -> int:
        """
        XOR 연산을 CNF로 인코딩 (Tseitin 변환).

        variables: [('var', v1), ('var', v2), ('const', 1), ...]
        returns: XOR 결과를 나타내는 변수 ID
        """
        if len(variables) == 0:
            # XOR() = 0
            result_var = self.var_counter
            self.var_counter += 1
            self.clauses.append([-result_var])  # result = False
            return result_var

        if len(variables) == 1:
            if variables[0][0] == 'const':
                # 상수 반환
                result_var = self.var_counter
                self.var_counter += 1
                if variables[0][1] == 1:
                    self.clauses.append([result_var])
                else:
                    self.clauses.append([-result_var])
                return result_var
            else:
                return variables[0][1]

        # 여러 변수의 XOR: 보조 변수로 체인 구성
        # r = a XOR b XOR c ...
        # r1 = a XOR b
        # r2 = r1 XOR c
        # ...

        current = None
        const_parity = 0

        for var_type, var_val in variables:
            if var_type == 'const':
                const_parity ^= var_val
            else:
                if current is None:
                    current = var_val
                else:
                    # current = current XOR var_val
                    new_var = self.var_counter
                    self.var_counter += 1
                    self._add_xor_clauses(current, var_val, new_var)
                    current = new_var

        # 최종 상수 parity 적용
        if const_parity == 1:
            if current is None:
                current = self.var_counter
                self.var_counter += 1
                self.clauses.append([current])
            else:
                # current = NOT current
                new_var = self.var_counter
                self.var_counter += 1
                self.clauses.append([-current, new_var])
                self.clauses.append([current, -new_var])
                current = new_var
        elif current is None:
            current = self.var_counter
            self.var_counter += 1
            self.clauses.append([-current])

        return current

    def _add_xor_clauses(self, a: int, b: int, result: int):
        """
        result = a XOR b 를 CNF 절로 추가.

        진리표:
        a | b | r
        0 | 0 | 0
        0 | 1 | 1
        1 | 0 | 1
        1 | 1 | 0

        CNF: (¬a ∨ ¬b ∨ ¬r) ∧ (a ∨ b ∨ ¬r) ∧ (a ∨ ¬b ∨ r) ∧ (¬a ∨ b ∨ r)
        """
        self.clauses.append([-a, -b, -result])
        self.clauses.append([a, b, -result])
        self.clauses.append([a, -b, result])
        self.clauses.append([-a, b, result])

    def _encode_at_least_k(self, variables: List[int], k: int):
        """
        At-Least-K 제약: variables 중 최소 k개가 True.

        Sequential Counter 인코딩 사용.
        """
        n = len(variables)
        if k > n:
            # 불가능한 제약: UNSAT을 만드는 빈 절 추가
            self.clauses.append([])
            return

        if k <= 0:
            # 항상 참: 절 추가 불필요
            return

        # Sequential Counter 인코딩
        # s[i][j] = "처음 i개 변수 중 j개 이상이 True"
        # 목표: s[n][k] = True

        s = {}  # (i, j) -> var_id

        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                s[(i, j)] = self.var_counter
                self.var_counter += 1

        # 초기 조건: s[1][1] ↔ x_1
        x1 = variables[0]
        self.clauses.append([-x1, s[(1, 1)]])
        self.clauses.append([x1, -s[(1, 1)]])

        # 재귀 관계
        for i in range(2, n + 1):
            xi = variables[i - 1]
            for j in range(1, min(i, k) + 1):
                sij = s[(i, j)]
                if j == 1:
                    # s[i][1] ↔ (s[i-1][1] ∨ x_i)
                    si1j = s.get((i - 1, 1))
                    if si1j:
                        self.clauses.append([-si1j, sij])
                        self.clauses.append([-xi, sij])
                        self.clauses.append([si1j, xi, -sij])
                    else:
                        # i=1, s[0][1] 없음
                        self.clauses.append([-xi, sij])
                        self.clauses.append([xi, -sij])
                elif j == i:
                    # s[i][i] ↔ (s[i-1][i-1] ∧ x_i)
                    si1j1 = s.get((i - 1, i - 1))
                    if si1j1:
                        self.clauses.append([sij, -si1j1, -xi])
                        self.clauses.append([si1j1, -sij])
                        self.clauses.append([xi, -sij])
                else:
                    # s[i][j] ↔ (s[i-1][j] ∨ (s[i-1][j-1] ∧ x_i))
                    si1j = s.get((i - 1, j))
                    si1j1 = s.get((i - 1, j - 1))

                    if si1j and si1j1:
                        self.clauses.append([-si1j, sij])
                        self.clauses.append([-si1j1, -xi, sij])
                        self.clauses.append([si1j, si1j1, -sij])
                        self.clauses.append([si1j, xi, -sij])

        # 최종 제약: s[n][k] = True
        if (n, k) in s:
            self.clauses.append([s[(n, k)]])

    def encode_rank_constraint(self):
        """
        Rank 제약: G의 행들이 선형 독립.

        최소 거리 제약이 이미 암묵적으로 포함하고 있으므로,
        선택적으로 추가 가능.
        """
        # 이 제약은 복잡하며, d_min >= 1 제약이 이미 rank = k를 보장함
        # 따라서 여기서는 구현 생략
        pass

    def add_symmetry_breaking_constraints(self):
        """
        대칭성 파괴 제약 추가로 탐색 공간 축소.

        예: 첫 번째 행의 패리티 부분 첫 비트를 1로 고정
        """
        if self.systematic and self.parity_bits > 0:
            # P[0, 0] = 1로 고정
            var = self.var_map[('P', 0, 0)]
            self.clauses.append([var])

    def export_to_dimacs(self, filename: str):
        """
        생성된 CNF를 DIMACS 형식 파일로 저장.

        형식:
        p cnf <변수 개수> <절 개수>
        <절1>
        <절2>
        ...
        """
        num_vars = self.var_counter - 1
        num_clauses = len(self.clauses)

        with open(filename, 'w') as f:
            f.write(f"p cnf {num_vars} {num_clauses}\n")
            for clause in self.clauses:
                f.write(' '.join(map(str, clause)) + ' 0\n')

        print(f"CNF 파일 생성 완료: {filename}")
        print(f"  변수 수: {num_vars}")
        print(f"  절 수: {num_clauses}")

    def encode_all_one_vector_constraint(self):
        """
        All-one vector 제약 조건 인코딩.

        For systematic form G = [I_k | P], the all-one vector (1,1,...,1)
        is in the code if and only if the XOR of all rows equals all-one.

        This is equivalent to:
        - For each column j: XOR of all G[i,j] = 1

        In systematic form:
        - Columns 0 to k-1 (identity part): Already satisfied (exactly one 1 per column)
        - Columns k to n-1 (parity part): Need to enforce XOR of P[:,j] = 1 for all j

        Mathematical background:
        - All-one ∈ C ⟺ Self-complementary ⟺ A_w = A_{n-w}
        - Many optimal codes satisfy this property

        References:
        - Pashinska-Gadzheva et al., Mathematics 11(24):4950, 2023
        - MacWilliams & Sloane, "Theory of Error-Correcting Codes", 1977
        """
        if not self.systematic:
            print("⚠️  All-one constraint only implemented for systematic form")
            return

        print(f"인코딩 중: All-one vector 제약 (Self-complementary property)")

        # For systematic form G = [I_k | P]:
        # Identity part [I_k]: Columns 0 to k-1
        #   Each column has exactly one 1 (at diagonal), so XOR = 1 ✓
        #   No constraint needed!

        # Parity part [P]: Columns k to n-1
        #   For each column j in P (0 <= j < n-k):
        #     XOR of P[0,j], P[1,j], ..., P[k-1,j] must equal 1

        parity_cols = self.n - self.k

        for j in range(parity_cols):
            # Collect variables for column j of P
            column_vars = []
            for i in range(self.k):
                var_id = self.var_map[('P', i, j)]
                column_vars.append(var_id)

            # Encode: XOR(column_vars) = 1
            # This means: odd number of True values in column

            # Use XOR chain to compute parity
            if len(column_vars) == 1:
                # Single variable: must be True
                self.clauses.append([column_vars[0]])
            else:
                # Multiple variables: XOR chain
                current = None

                for var in column_vars:
                    if current is None:
                        current = var
                    else:
                        # Create new variable for XOR result
                        new_var = self.var_counter
                        self.var_counter += 1
                        self._add_xor_clauses(current, var, new_var)
                        current = new_var

                # Final result must be True (= 1)
                self.clauses.append([current])

        print(f"  추가된 제약: {parity_cols}개 열에 대한 XOR = 1")

    def encode_all_constraints(self, include_all_one=False):
        """
        모든 제약 조건을 인코딩하는 메인 함수.

        Parameters
        ----------
        include_all_one : bool
            All-one vector 제약 포함 여부 (default: False)
        """
        print(f"\n{'='*60}")
        print(f"CNF 인코딩 시작: [{self.n}, {self.k}, {self.d_min}] 코드")
        print(f"Systematic form: {self.systematic}")
        print(f"All-one constraint: {include_all_one}")
        print(f"{'='*60}\n")

        # 최소 거리 제약 (가장 중요)
        self.encode_minimum_distance_constraints()

        # All-one vector 제약 (선택적)
        if include_all_one:
            self.encode_all_one_vector_constraint()

        # 대칭성 파괴 (선택적)
        self.add_symmetry_breaking_constraints()

        print(f"\n인코딩 완료!")
        print(f"총 변수 수: {self.var_counter - 1}")
        print(f"총 절 수: {len(self.clauses)}")


if __name__ == "__main__":
    # 예제: [7, 4, 3] 해밍 코드로 테스트
    print("예제: [7, 4, 3] 해밍 코드 인코딩\n")
    encoder = CNFEncoder(n=7, k=4, d_min=3, systematic=True)
    encoder.encode_all_constraints()
    encoder.export_to_dimacs("hamming_7_4_3.cnf")

    print("\n" + "="*60)
    print("예제: [16, 7, 6] 코드 인코딩")
    print("="*60 + "\n")
    encoder2 = CNFEncoder(n=16, k=7, d_min=6, systematic=True)
    encoder2.encode_all_constraints()
    encoder2.export_to_dimacs("code_16_7_6.cnf")
