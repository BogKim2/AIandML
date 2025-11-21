# Week 14: PINN 응용 I - 편미분방정식 (Partial Differential Equations)

## 📚 학습 목표

이번 주차에서는 **Physics-Informed Neural Networks (PINN)**을 사용하여 편미분방정식(PDE)을 푸는 방법을 학습합니다.

**배울 내용:**
1. PINN의 기본 개념과 작동 원리
2. 1D/2D Heat Equation (열전도 방정식)
3. 1D/2D Wave Equation (파동 방정식)
4. Burgers Equation (비선형 PDE)
5. 복잡한 경계조건 처리

**왜 중요한가?**
- 전통적 수치해법(FEM, FDM)의 한계를 극복
- 데이터가 부족한 상황에서도 물리 법칙으로 학습 가능
- 역문제(Inverse Problems) 해결에 유용
- 실시간 시뮬레이션 및 최적화에 활용

---

## 🎯 PINN이란?

### 정의

**Physics-Informed Neural Networks (PINN)**는 신경망이 데이터와 함께 **물리 법칙(PDE)**을 직접 학습하도록 하는 기법입니다.

### 전통적 방법 vs PINN

```
[전통적 수치해법]
PDE → 이산화 (격자) → 행렬 방정식 → 반복 해법
단점: 격자 의존적, 고차원에서 비효율적

[PINN]
PDE → 신경망 손실함수 → 자동미분 → 최적화
장점: 격자 불필요, 고차원 확장 용이, 역문제 해결 가능
```

### PINN의 핵심 아이디어

1. **근사 함수**: 신경망 $u_{\theta}(x, t)$로 해 $u(x, t)$를 근사
2. **자동 미분**: TensorFlow/PyTorch의 `GradientTape`로 $\frac{\partial u}{\partial x}$, $\frac{\partial^2 u}{\partial x^2}$ 계산
3. **물리 법칙 손실**: PDE 잔차를 손실함수에 포함
4. **경계/초기조건 손실**: 추가 제약 조건으로 학습 안정화

---

## 🔬 Lab 1: PINN 기본 - 간단한 ODE (01_basic_pinn.py)

### 목적
가장 간단한 미분방정식 $\frac{du}{dt} = -u$를 PINN으로 풀어보며 기본 개념을 이해합니다.

### 문제 설정

**미분방정식:**
$$\frac{du}{dt} = -u$$

**초기조건:**
$$u(0) = 1$$

**해석해 (정답):**
$$u(t) = e^{-t}$$

### 프로그램 실행

```bash
cd week14
uv run 01_basic_pinn.py
```

### 핵심 코드 설명

**1. 물리 법칙 손실 함수:**
```python
@tf.function
def physics_loss(t):
    with tf.GradientTape() as tape:
        tape.watch(t)
        u = model(t)
    du_dt = tape.gradient(u, t)  # 자동 미분!
    
    # ODE: du/dt + u = 0
    pde_residual = du_dt + u
    return tf.reduce_mean(tf.square(pde_residual))
```

**핵심 포인트:**
- `GradientTape`: 자동으로 $\frac{du}{dt}$를 계산
- `pde_residual`: PDE가 0이 되도록 강제
- 잔차의 제곱을 최소화 → PDE를 만족하는 해 찾기

**2. 초기조건 손실:**
```python
def initial_loss(t_init, u_init):
    u_pred = model(t_init)
    return tf.reduce_mean(tf.square(u_pred - u_init))
```

**3. 전체 손실:**
```python
total_loss = physics_loss(t_physics) + 10.0 * initial_loss(t_init, u_init)
```

초기조건에 큰 가중치를 주어 $u(0) = 1$을 정확히 만족하도록 합니다.

### 결과

- **MSE**: ~10^-6 수준 (해석해와 거의 일치)
- **핵심 통찰**: 데이터 없이 물리 법칙만으로 미분방정식을 풀 수 있다!

---

## 🔬 Lab 2: 1D Heat Equation (02_heat_equation_1d.py)

### 목적
1차원 열전도 방정식을 PINN으로 풀고, 시간에 따른 온도 분포를 시각화합니다.

### 문제 설정

**열전도 방정식:**
$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$

- $u(x, t)$: 위치 $x$, 시간 $t$에서의 온도
- $\alpha$: 열확산계수 (0.01)

**초기조건:**
$$u(x, 0) = \sin(\pi x)$$

**경계조건:**
$$u(0, t) = u(1, t) = 0$$

**해석해:**
$$u(x, t) = \sin(\pi x) e^{-\alpha \pi^2 t}$$

### 프로그램 실행

```bash
uv run 02_heat_equation_1d.py
```

### 핵심 개념

**1. 2차 미분 계산:**
```python
@tf.function
def pde_residual(x, t):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, t])
        u = model(tf.concat([x, t], axis=1))
        
        du_dt = tape.gradient(u, t)      # ∂u/∂t
        du_dx = tape.gradient(u, x)      # ∂u/∂x
    
    d2u_dx2 = tape.gradient(du_dx, x)    # ∂²u/∂x²
    
    # Heat equation residual
    residual = du_dt - alpha * d2u_dx2
    return residual
```

**핵심:**
- `persistent=True`: 여러 번 미분하기 위해 필요
- 중첩된 `GradientTape`: 2차 미분 계산

**2. 경계조건:**
```python
def boundary_condition_loss(x_bc, t_bc):
    u_pred = model(tf.concat([x_bc, t_bc], axis=1))
    u_true = tf.zeros_like(u_pred)  # u=0 at boundaries
    return tf.reduce_mean(tf.square(u_pred - u_true))
```

### 물리적 의미

- **초기**: 사인파 온도 분포
- **시간 경과**: 경계(x=0, x=1)가 0도로 고정되어 있어 열이 빠져나감
- **최종**: 모든 곳의 온도가 0도로 수렴

### 결과 분석

- **3D 플롯**: 시공간에서의 온도 변화
- **시간 스냅샷**: 특정 시간 $t$에서의 온도 분포
- **오차**: 해석해와 비교하여 PINN의 정확도 확인

---

## 🔬 Lab 3: 1D Wave Equation (03_wave_equation_1d.py)

### 목적
1차원 파동 방정식을 풀어 진동하는 현의 움직임을 시뮬레이션합니다.

### 문제 설정

**파동 방정식:**
$$\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}$$

- $u(x, t)$: 변위
- $c$: 파동 속도 (1.0)

**초기조건:**
$$u(x, 0) = \sin(\pi x), \quad \frac{\partial u}{\partial t}(x, 0) = 0$$

**경계조건:**
$$u(0, t) = u(1, t) = 0$$

**해석해:**
$$u(x, t) = \sin(\pi x) \cos(\pi c t)$$

### 프로그램 실행

```bash
uv run 03_wave_equation_1d.py
```

### 핵심 차이점

Wave equation은 **2차 시간 미분**을 포함합니다:

```python
with tf.GradientTape(persistent=True) as tape2:
    with tf.GradientTape(persistent=True) as tape1:
        u = model(xt)
    du_dt = tape1.gradient(u, t)

d2u_dt2 = tape2.gradient(du_dt, t)  # ∂²u/∂t²
```

**초기 속도 조건:**
```python
def initial_velocity_loss(x_ic, t_ic):
    with tf.GradientTape() as tape:
        tape.watch(t_ic)
        u = model(tf.concat([x_ic, t_ic], axis=1))
    du_dt = tape.gradient(u, t_ic)
    # ∂u/∂t(x, 0) = 0
    return tf.reduce_mean(tf.square(du_dt))
```

### 물리적 의미

- **t=0**: 사인파 모양으로 변위, 속도는 0 (정지 상태)
- **진동**: 경계가 고정되어 있어 정상파(standing wave) 형성
- **주기성**: $T = \frac{2}{c} = 2$ 초마다 원래 모양으로 복귀

---

## 🔬 Lab 4: 2D Heat Equation (04_heat_equation_2d.py)

### 목적
2차원 평면에서의 열전도를 시뮬레이션하여 고차원 PDE 해법을 학습합니다.

### 문제 설정

**2D 열전도 방정식:**
$$\frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)$$

**초기조건:**
$$u(x, y, 0) = \sin(\pi x) \sin(\pi y)$$

**경계조건:**
$$u = 0 \text{ on all boundaries}$$

### 프로그램 실행

```bash
uv run 04_heat_equation_2d.py
```

### 고차원의 도전

**1. 입력 차원 증가:**
- 1D: $(x, t)$ - 2개 입력
- 2D: $(x, y, t)$ - 3개 입력

**2. 경계조건 복잡도:**
- 4개 경계 (x=0, x=1, y=0, y=1) 각각 처리

**3. 훈련 데이터 증가:**
- 2D 공간을 충분히 샘플링하려면 더 많은 점 필요
- 경계조건 점도 4배 증가

### 시각화

- **Contour Plot**: 온도 분포를 등고선으로 표현
- **3D Surface**: 열 확산 과정을 3차원으로 시각화
- **시간별 스냅샷**: t=0, 0.1, 0.3, 0.5에서의 온도 분포 비교

---

## 🔬 Lab 5: Burgers Equation (05_burgers_equation.py)

### 목적
**비선형** PDE의 대표적 예제인 Burgers 방정식을 풀어봅니다.

### 문제 설정

**Burgers 방정식:**
$$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}$$

- **비선형 항**: $u \frac{\partial u}{\partial x}$ (대류 항)
- **확산 항**: $\nu \frac{\partial^2 u}{\partial x^2}$
- $\nu$: 점성 계수 (0.01/π)

**초기조건:**
$$u(x, 0) = -\sin(\pi x)$$

**경계조건:**
$$u(-1, t) = u(1, t) = 0$$

### 프로그램 실행

```bash
uv run 05_burgers_equation.py
```

### 비선형 PDE의 도전

**1. 비선형 항 처리:**
```python
residual = du_dt + u * du_dx - nu * d2u_dx2
#                  ↑ 비선형!
```

**핵심:**
- $u$와 $\frac{\partial u}{\partial x}$의 **곱**
- 해가 자기 자신에 영향을 미침 → 복잡한 동역학

**2. 훈련 어려움:**
- 선형 PDE보다 수렴이 느림
- 더 많은 epoch 필요 (10,000회)
- 학습률 조정 중요

### 물리적 의미

Burgers 방정식은:
- **Navier-Stokes의 간단한 버전** (1D, 압축성 없음)
- **충격파(Shock Wave)** 형성 가능
- 유체역학의 기초

---

## 🔬 Lab 6: 2D Wave Equation (06_wave_equation_2d.py)

### 목적
2차원 막(membrane)의 진동을 시뮬레이션합니다.

### 문제 설정

**2D 파동 방정식:**
$$\frac{\partial^2 u}{\partial t^2} = c^2 \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)$$

**초기조건:**
$$u(x, y, 0) = \sin(\pi x) \sin(\pi y), \quad \frac{\partial u}{\partial t}(x, y, 0) = 0$$

**경계조건:**
$$u = 0 \text{ on all boundaries}$$

### 프로그램 실행

```bash
uv run 06_wave_equation_2d.py
```

### 복잡도

- **가장 높은 차원**: $(x, y, t)$ - 3개 입력
- **2차 시간 미분 + 2차 공간 미분** - 총 3개의 2차 미분
- **긴 훈련 시간**: 8,000 epochs

### 물리적 의미

- **고정된 경계**: 사각형 막의 테두리가 고정
- **진동 모드**: 특정 주파수로 진동하는 정상파
- **실용 예**: 드럼, 스피커 진동판

---

## 🔬 Lab 7: 복잡한 경계조건 (07_complex_boundary.py)

### 목적
실제 문제에서 자주 등장하는 **복잡한 기하학**과 **혼합 경계조건**을 다룹니다.

### 문제 설정

**영역**: L자 모양 (L-shaped domain)
- $(0 \leq x \leq 1, 0 \leq y \leq 1)$에서
- 오른쪽 위 코너 $(0.5 \leq x \leq 1, 0.5 \leq y \leq 1)$ 제외

**경계조건 (혼합):**
1. **Dirichlet BC** (x=0): $u = 0$ (고정 온도)
2. **Neumann BC** (y=0): $\frac{\partial u}{\partial n} = 0$ (단열)

**초기조건:**
$$u(x, y, 0) = e^{-50((x-0.25)^2 + (y-0.25)^2)}$$
(중앙에 뜨거운 점)

### 프로그램 실행

```bash
uv run 07_complex_boundary.py
```

### 핵심 기법

**1. 영역 정의:**
```python
def is_in_domain(x, y):
    """L자 영역 확인"""
    in_domain = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
    excluded = (x >= 0.5) & (x <= 1) & (y >= 0.5) & (y <= 1)
    return in_domain & ~excluded
```

**2. Neumann BC (법선 미분):**
```python
@tf.function
def neumann_bc_loss(x_bc, y_bc, t_bc, nx, ny):
    with tf.GradientTape() as tape:
        tape.watch([x_bc, y_bc])
        u = model(tf.concat([x_bc, y_bc, t_bc], axis=1))
    
    du_dx = tape.gradient(u, x_bc)
    du_dy = tape.gradient(u, y_bc)
    
    # 법선 방향 미분: ∂u/∂n = ∇u · n
    du_dn = du_dx * nx + du_dy * ny
    return tf.reduce_mean(tf.square(du_dn))
```

**핵심:**
- `nx, ny`: 법선 벡터 성분
- $\frac{\partial u}{\partial n} = \nabla u \cdot \mathbf{n}$
- Neumann BC는 "열이 경계를 통해 나가지 않는다" (단열)

### 실용적 의미

- **실제 공학 문제**: 대부분 복잡한 기하학
- **PINN의 강점**: 격자 없이 임의의 형상 처리 가능
- **전통적 방법**: 격자 생성이 매우 어려움

---

## 📊 PINN vs 전통적 방법 비교

| 특성 | PINN | FEM/FDM |
|------|------|---------|
| **격자 의존성** | 불필요 | 필수 |
| **고차원 확장** | 용이 | 어려움 (차원의 저주) |
| **경계조건** | 손실함수로 자연스럽게 포함 | 격자에 강제 부과 |
| **역문제** | 가능 (매개변수 추정) | 어려움 |
| **정확도** | 해석해 대비 ~10^-3 ~ 10^-6 | ~10^-6 ~ 10^-8 |
| **계산 비용** | GPU 활용 시 빠름 | 대규모 행렬 연산 |

---

## 🎓 핵심 개념 정리

### 1. PINN의 3대 손실 함수

```python
total_loss = loss_pde + λ_ic * loss_ic + λ_bc * loss_bc
```

- **PDE 손실**: 물리 법칙 만족
- **초기조건 손실**: $t=0$에서의 상태
- **경계조건 손실**: 영역 경계에서의 제약

### 2. 자동 미분의 마법

TensorFlow의 `GradientTape`는:
- 임의 차수의 미분 계산 가능
- 복잡한 PDE도 자동으로 처리
- 역전파(backpropagation)로 최적화

### 3. 하이퍼파라미터 선택

**중요한 설정:**
1. **신경망 크기**: 복잡한 PDE일수록 더 깊고 넓게
2. **손실 가중치**: 초기/경계조건에 큰 가중치 (10~20)
3. **샘플링**: 경계와 초기조건에 충분한 점
4. **학습률**: 비선형 PDE는 작게 (0.001)
5. **Epochs**: 최소 5,000 ~ 10,000

---

## 🚀 다음 단계

### Week 15 예고: PINN 응용 II
- Navier-Stokes 방정식 (유체역학)
- 역문제 (Inverse Problems)
- 매개변수 추정
- 데이터 동화 (Data Assimilation)

### 추가 학습 자료
1. **논문**: "Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations" (Raissi et al., 2019)
2. **코드**: DeepXDE, PINN 전용 라이브러리
3. **응용 분야**: 재료 과학, 지진학, 약물 전달 시스템

---

## 📝 학습 체크리스트

- [ ] PINN의 기본 개념 이해
- [ ] 자동 미분으로 PDE 잔차 계산 가능
- [ ] 1D Heat/Wave Equation 실행 및 결과 해석
- [ ] 2D PDE의 복잡도 이해
- [ ] 비선형 PDE (Burgers) 풀이
- [ ] Dirichlet vs Neumann 경계조건 구분
- [ ] 복잡한 기하학 처리 방법 이해
- [ ] PINN의 장단점 설명 가능

---

## 🛠️ 실습 Tips

### 훈련이 잘 안 될 때

1. **손실이 감소하지 않음**
   - 학습률 줄이기 (0.001 → 0.0001)
   - 초기/경계조건 가중치 높이기

2. **진동하는 손실**
   - Adam optimizer의 β 값 조정 (β1=0.9, β2=0.999)
   - 배치 크기 늘리기

3. **경계조건 위반**
   - 경계 샘플 수 증가
   - 경계 손실 가중치 극대화 (100 이상)

### 디버깅

```python
# 각 손실 성분 모니터링
if (epoch + 1) % 100 == 0:
    print(f"PDE: {loss_pde:.6f}, IC: {loss_ic:.6f}, BC: {loss_bc:.6f}")
```

- PDE 손실이 크면: 신경망 용량 부족 → 레이어 추가
- IC/BC 손실이 크면: 가중치 증가

---

## 📚 참고 자료 (References)

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *Journal of Computational Physics*, 378, 686-707.

2. Karniadakis, G. E., et al. (2021). "Physics-informed machine learning." *Nature Reviews Physics*, 3(6), 422-440.

3. TensorFlow Documentation: [Automatic Differentiation](https://www.tensorflow.org/guide/autodiff)

---

**다음 주 준비**: Navier-Stokes 방정식에 대해 미리 학습해보세요!
