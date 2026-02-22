# Lambda_15(1^10) Perfect Partition - 전략 계획

## 핵심 데이터
- |V| = 32,527
- Ball sizes: {14: 435, 15: 1305, 16: 30787}
- Expected centers: 2042.26
- 최소 가능 centers: 2033 (2032×16 + 1×15)

## 이전 실패 분석

### 1. Full SAT (8h timeout) - 실패
- CNF: 518K vars, 3.4M clauses
- 문제: 탐색 공간이 너무 넓음

### 2. s=11 Repair - UNSAT
- 11개 bad center 교체 시도
- 문제: 교체 후보가 모두 기존 center와 거리 < 3

### 3. Greedy + SAT - UNSAT
- Greedy로 ~80% 커버 후 SAT로 완성 시도
- 문제: Greedy가 "나쁜 영역"을 선점해 나머지 커버 불가

### 4. MaxSAT Seed - 미완성
- Covering만 soft clause, overlap 허용
- 문제: Perfect partition이 아닌 cover만 찾음

## 새로운 접근법

### Strategy A: Local Search with Restart (Simulated Annealing)
- 초기 해: 무작위 greedy로 가능한 많이 선택
- Move: center 교체/추가/제거
- Energy: uncovered + overlap count
- 장점: SAT의 탐색 공간 문제 우회

### Strategy B: Constraint Propagation + Backtracking
- Ball size 14/15 정점 우선 처리 (선택지 제한)
- 이들의 커버 방법이 결정되면 나머지는 더 쉬움
- Arc consistency로 가지치기

### Strategy C: ILP with Column Generation
- 변수: 각 center 선택 여부
- 제약: 각 정점 정확히 1회 커버
- Column generation으로 점진적 해결

### Strategy D: Genetic Algorithm with Smart Crossover
- 개체: center 집합
- Fitness: coverage - overlap penalty
- Crossover: 공간적으로 분리된 영역 조합

### Strategy E: Divide and Conquer
- 그래프를 여러 영역으로 분할
- 각 영역 독립적으로 해결 후 병합
- 경계 처리가 관건

### Strategy F: Algebraic Construction
- Hamming code 기반 변형
- Cyclic/circulant 구조 활용
- 수학적 구조로 탐색 공간 축소

## 권장 우선순위

1. **Strategy A (Local Search)**: 구현 쉽고 빠른 피드백
2. **Strategy B (Constraint Propagation)**: ball size 14/15 먼저 처리
3. **Strategy F (Algebraic)**: 장기적으로 가장 elegant

## 즉시 실행 계획

### Phase 1: Ball size 14/15 분석
- 이 정점들의 분포와 상호 관계 분석
- 이들이 반드시 center가 되어야 하는지, 아니면 인접 center에 의해 커버 가능한지

### Phase 2: Local Search 구현
- Greedy initial solution
- Swap/Add/Remove moves
- Tabu list로 cycling 방지

### Phase 3: 결과 분석 및 반복
