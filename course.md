# 전산물리: AI와 물리학의 만남
## Computational Physics: From Neural Networks to Physics-Informed AI

**부산대학교 물리학과**  
**학년**: 3학년  
**학기**: 2025년 1학기  
**강의시간**: 주 3시간 (강의 2시간 + 실습 1시간)  
**사용 도구**: Python, Claude AI, VS Code  

---

## 📚 강의 개요

본 강의는 현대 물리학 연구에서 필수적인 인공지능 기술을 학습하고, 이를 실제 물리 문제 해결에 적용하는 능력을 배양합니다. Neural Network의 기초부터 최신 Large Language Model(LLM), 그리고 물리학 특화 AI인 Physics-Informed Neural Networks(PINN)까지 다룹니다.

**핵심 특징**:
- MIT 강의 자료 기반의 체계적인 이론 학습
- Claude AI를 활용한 "vibe coding" 실습
- 역학, 전자기, 양자역학, 통계물리 문제의 AI 기반 수치 해법
- 물리 법칙을 학습에 직접 포함하는 PINN 기법

---

## 🎯 학습 목표

1. Neural Network의 기본 원리와 작동 방식 이해
2. Deep Learning과 Transformer 아키텍처 학습
3. LLM을 활용한 효율적인 코딩 능력 배양
4. 물리 문제를 AI로 해결하는 실전 경험
5. PINN을 이용한 미분방정식 해법 습득

---

## 📖 교재 및 참고자료

### 주교재
- **MIT 6.S191**: Introduction to Deep Learning (강의 노트 및 영상)
- 강의자 제공 자료: Jupyter Notebooks, Python 코드 예제

### 참고서적
- *Deep Learning* by Goodfellow, Bengio, and Courville
- *Physics-Informed Neural Networks* (관련 논문 모음)
- *Computational Physics* by Mark Newman

### 온라인 자료
- Claude AI Documentation
- PyTorch/TensorFlow Tutorials
- ArXiv papers on PINN applications

---

## 📅 강의 일정 (16주)

### 🔷 Part I: Neural Networks & Deep Learning (Weeks 1-7)

#### **Week 1: 강의 소개 및 환경 설정**
- 강의 목표 및 평가 방식 소개
- Python 환경 설정: Anaconda, VS Code, Git
- Claude AI 계정 생성 및 기본 사용법
- **실습**: "Hello, Neural Network!" - 첫 번째 신경망 구현

**주요 개념**: Development Environment, AI in Physics

---

#### **Week 2: 머신러닝 기초**
- 머신러닝의 세 가지 유형: 지도/비지도/강화 학습
- 데이터 전처리와 특징 추출
- 손실 함수(Loss Function)와 최적화
- **실습**: Linear Regression으로 물리 데이터 피팅

**주요 개념**: Supervised Learning, Loss Functions, Gradient Descent

**과제 1**: 실험 데이터를 이용한 선형/비선형 회귀 분석

---

#### **Week 3: Neural Network 기초 이론**
- Perceptron과 Multi-Layer Perceptron
- Activation Functions: ReLU, Sigmoid, Tanh
- Forward Propagation의 수학적 구조
- Universal Approximation Theorem
- **실습**: Numpy로 구현하는 순수 Neural Network

**주요 개념**: Neurons, Activation, Forward Pass

**참고**: MIT 6.S191 Lecture 1

---

#### **Week 4: Backpropagation과 최적화**
- Backpropagation 알고리즘의 원리
- Chain Rule과 자동 미분
- Gradient Descent 변형: SGD, Momentum, Adam
- Learning Rate Scheduling
- **실습**: 역전파 알고리즘 직접 구현하기

**주요 개념**: Backpropagation, Optimizers, Gradient Flow

**과제 2**: 다양한 Optimizer 성능 비교 실험

---

#### **Week 5: Deep Learning의 핵심 기법**
- Regularization: L1/L2, Dropout, Batch Normalization
- Overfitting vs. Underfitting
- Data Augmentation
- Transfer Learning의 개념
- **실습**: MNIST 손글씨 인식 (CNN 도입)

**주요 개념**: Regularization, Generalization, CNN Basics

**참고**: MIT 6.S191 Lecture 2

---

#### **Week 6: Transformer와 Attention Mechanism**
- RNN의 한계와 Attention의 등장
- Self-Attention의 원리
- Transformer 아키텍처 분석
- Positional Encoding
- **실습**: 간단한 Sequence Modeling

**주요 개념**: Attention, Transformers, Sequence Modeling

**참고**: "Attention Is All You Need" paper

---

#### **Week 7: Large Language Models (LLM) 개론**
- GPT, BERT, Claude 아키텍처 비교
- Pre-training과 Fine-tuning
- Token, Embedding, Context Window
- LLM의 물리학 응용 가능성
- **실습**: Claude API를 이용한 자동 코드 생성

**주요 개념**: LLM Architecture, Tokenization, Prompting

**프로젝트 중간 발표**: Part I 학습 내용 요약 및 미니 프로젝트

---

### 🔷 Part II: LLM Vibe Coding for Physics (Weeks 8-12)

#### **Week 8: 중간고사 / LLM 기반 코딩 입문**
- "Vibe Coding"이란? - 자연어로 코드 생성
- Prompt Engineering 기법
- Claude를 이용한 효율적인 디버깅
- **실습**: 자연어로 물리 시뮬레이션 코드 생성

**주요 개념**: Prompt Engineering, AI-Assisted Programming

---

#### **Week 9: 고전 역학 문제 해결**
- 뉴턴 방정식의 수치 해법 (Euler, RK4)
- 행성 운동 시뮬레이션
- 진자 운동과 혼돈 (Chaotic Systems)
- 라그랑지안과 해밀토니안 역학
- **실습**: 
  - 이중 진자(Double Pendulum) 시뮬레이션
  - 3체 문제 수치 해법
  - LLM으로 자동 코드 생성 및 최적화

**주요 개념**: ODEs, Numerical Integration, Chaotic Dynamics

**과제 3**: LLM을 활용한 역학 문제 풀이 및 시각화

---

#### **Week 10: 전자기학 시뮬레이션**
- Maxwell 방정식의 수치 해법
- 전기장/자기장 계산 및 시각화
- 유한 차분법(Finite Difference Method)
- 전자기파 전파 시뮬레이션
- **실습**:
  - 다중 점전하의 전기장 계산
  - 전자기파 전파 애니메이션
  - 도체 내부 전위 분포 계산

**주요 개념**: Maxwell Equations, FDM, Vector Field Visualization

**과제 4**: 복잡한 전하 배치의 전기장 분석

---

#### **Week 11: 양자역학 시뮬레이션**
- Schrödinger 방정식의 수치 해법
- 파동함수 시각화
- 터널링 효과 시뮬레이션
- Finite Well, Harmonic Oscillator
- **실습**:
  - 1차원 포텐셜 우물 문제
  - 시간 의존 Schrödinger 방정식
  - 파동 패킷의 시간 발전

**주요 개념**: Quantum Mechanics, Wave Functions, Numerical Eigenvalue Problems

**과제 5**: 다양한 포텐셜에서의 양자 상태 분석

---

#### **Week 12: 통계물리 및 Monte Carlo 시뮬레이션**
- Monte Carlo 방법론
- Ising 모델과 상전이
- Metropolis-Hastings 알고리즘
- 열역학적 성질 계산
- **실습**:
  - 2D Ising 모델 시뮬레이션
  - 상전이 온도 계산
  - Partition Function 추정

**주요 개념**: Statistical Physics, Monte Carlo, Phase Transitions

**프로젝트 중간 점검**: Part II 실습 결과 공유

---

### 🔷 Part III: Physics-Informed Neural Networks (Weeks 13-14)

#### **Week 13: PINN 기초 이론**
- Physics-Informed Neural Networks 개념
- 미분방정식을 Loss Function에 포함하기
- Automatic Differentiation
- PINN vs. 전통적 수치 해법
- **실습**: 
  - 간단한 ODE를 PINN으로 풀기
  - 경계조건 처리 방법

**주요 개념**: PINN, Physics Loss, Boundary Conditions

**참고 논문**: 
- Raissi et al., "Physics-informed neural networks" (2019)
- Karniadakis et al., "Physics-informed machine learning" (2021)

---

#### **Week 14: PINN 응용 I - 편미분방정식**
- 열전도 방정식 (Heat Equation)
- 파동 방정식 (Wave Equation)
- Burgers 방정식
- Navier-Stokes 방정식 (개념)
- **실습**:
  - 1D Heat Equation 풀이
  - 2D Wave Equation 시뮬레이션
  - 복잡한 경계조건 처리

**주요 개념**: PDEs, PINN for Spatial-Temporal Problems

**과제 6**: PINN을 이용한 편미분방정식 해법 프로젝트

---

### 🔷 Part IV: 최종 프로젝트 (Weeks 15-16)

#### **Week 15: PINN 응용 II - 고급 주제 & 최종 프로젝트 시작**
- Inverse Problems with PINN
- Multi-fidelity Learning
- Transfer Learning in PINN
- 현대 물리학 연구에서의 AI 활용 사례
- **최종 프로젝트 시작**: 팀별 주제 선정 및 계획 수립

**프로젝트 주제 예시**:
1. PINN을 이용한 Schrödinger 방정식 풀이
2. 복잡한 경계조건의 전자기 문제 해결
3. 상전이 시뮬레이션에 AI 적용
4. 실험 데이터 기반 물리량 예측
5. 역문제(Inverse Problem) 해결

---

#### **Week 16: 최종 프로젝트 발표 및 기말고사**
- 팀별 프로젝트 결과 발표 (20분)
- 코드 리뷰 및 피드백
- 기말 시험 (이론 + 실습 문제)
- 강의 총평 및 향후 학습 방향 제시

---

## 📊 평가 방식

| 항목 | 비율 | 세부 내용 |
|------|------|----------|
| **과제** | 30% | 주간 과제 6회 (각 5%) |
| **중간 프로젝트** | 15% | Part I 요약 발표 |
| **최종 프로젝트** | 25% | 팀 프로젝트 (코드 + 보고서 + 발표) |
| **중간고사** | 10% | 이론 및 기본 실습 |
| **기말고사** | 15% | 종합 평가 |
| **출석 및 참여** | 5% | 수업 참여도 및 토론 |

### 과제 제출 방식
- GitHub Repository를 통한 코드 제출
- Jupyter Notebook 형식 권장
- README.md에 실행 방법 및 결과 분석 포함

### 최종 프로젝트 요구사항
- 2-3명 팀 구성
- GitHub을 통한 협업
- 10-15페이지 보고서 (LaTeX 권장)
- 20분 발표 + 10분 질의응답
- 재현 가능한 코드 및 데이터

---

## 💻 실습 환경

### 필수 소프트웨어
```bash
# Python 3.9 이상
conda create -n compphys python=3.10
conda activate compphys

# 필수 패키지
pip install numpy scipy matplotlib
pip install pandas seaborn
pip install jupyter notebook
pip install torch torchvision  # PyTorch
pip install anthropic  # Claude API

# 선택 패키지
pip install plotly  # 인터랙티브 시각화
pip install sympy  # 심볼릭 계산
```

### VS Code 확장 프로그램
- Python
- Jupyter
- GitHub Copilot (선택)
- LaTeX Workshop (보고서 작성용)

### Claude AI 설정
- Claude.ai 계정 생성
- API Key 발급 (실습용)
- Prompt Engineering 실습

---

## 📚 주차별 읽기 자료

### Week 1-2
- Newman, "Computational Physics", Chapter 1-2
- Python for Physics 튜토리얼

### Week 3-5
- MIT 6.S191 Lecture Notes 1-3
- Goodfellow et al., "Deep Learning", Chapter 6

### Week 6-7
- "Attention Is All You Need" - Vaswani et al. (2017)
- GPT-3 Paper (Brown et al., 2020)

### Week 9-12
- Newman, "Computational Physics", Chapter 8 (ODEs)
- Landau & Lifshitz, Classical Mechanics (참조)

### Week 13-14
- Raissi et al., "Physics-informed neural networks" (2019)
- Karniadakis et al., "Physics-informed machine learning" (2021)
- Cuomo et al., "Scientific Machine Learning" (2022)

---

## 🎓 선수 과목

- **필수**: 
  - 일반물리학 I, II
  - 역학
  - 전자기학 I
  - Python 프로그래밍 기초

- **권장**:
  - 양자역학 I
  - 수리물리학
  - 통계물리

---

## 🔗 유용한 링크

### 강의 자료
- [MIT 6.S191 Course Website](http://introtodeeplearning.com/)
- [Physics-Informed Neural Networks GitHub](https://github.com/maziarraissi/PINNs)

### 온라인 리소스
- [Physics-Informed DeepONet](https://www.sciencedirect.com/science/article/pii/S0021999121005787)
- [DeepXDE Library](https://deepxde.readthedocs.io/)

### 논문 저장소
- [ArXiv: Physics + AI](https://arxiv.org/list/physics.comp-ph/recent)
- [ArXiv: Machine Learning](https://arxiv.org/list/cs.LG/recent)

---

## 📝 과제 상세 정보

### 과제 1: 데이터 피팅 (Week 2)
실험 데이터를 제공하고, 다양한 회귀 모델로 피팅 후 최적 모델 선택

### 과제 2: Optimizer 비교 (Week 4)
SGD, Momentum, Adam 등의 optimizer를 동일한 문제에 적용하여 성능 비교

### 과제 3: 역학 시뮬레이션 (Week 9)
LLM을 활용하여 복잡한 역학 문제 시뮬레이션 코드 작성

### 과제 4: 전자기 시각화 (Week 10)
주어진 전하 배치의 전기장을 계산하고 아름다운 시각화 생성

### 과제 5: 양자 시뮬레이션 (Week 11)
다양한 포텐셜에서의 양자 상태 계산 및 분석

### 과제 6: PINN 프로젝트 (Week 14)
선택한 편미분방정식을 PINN으로 풀고 전통 방법과 비교

---

## 🌟 학습 성과

본 강의를 성공적으로 이수한 학생은:

1. ✅ Neural Network의 작동 원리를 수학적으로 이해
2. ✅ LLM을 활용한 효율적인 코딩 능력 보유
3. ✅ 물리 문제를 AI로 해결하는 실전 경험
4. ✅ PINN을 이용한 미분방정식 해법 습득
5. ✅ 최신 AI 기술의 물리학 응용 능력
6. ✅ GitHub을 통한 협업 및 코드 관리 능력
7. ✅ 연구 논문 작성 및 발표 능력

---

## 💡 강의 철학

> "물리학자는 자연을 이해하는 사람이며,  
> 계산물리학자는 자연을 시뮬레이션하는 사람이고,  
> AI 시대의 물리학자는 자연을 학습하고 예측하는 사람입니다."

본 강의는 단순히 AI 도구 사용법을 가르치는 것이 아니라,
**물리적 직관과 AI 기술을 융합하여 새로운 문제 해결 방식**을 
배우는 것을 목표로 합니다.

---

## 📧 연락처 및 Office Hours

**강의실**: TBA  
**실습실**: TBA  
**Office Hours**: 수요일 14:00-16:00 (사전 예약 권장)  
**이메일**: TBA  
**강의 GitHub**: TBA  
**Q&A**: Slack 채널 운영

---

## 📌 주의사항

1. **학술 윤리**: AI 도구 사용 시 출처를 명확히 밝혀야 합니다
2. **코드 공유**: 과제는 개인 작업이며, 코드 복사는 금지됩니다
3. **출석**: 실습 중심 강의이므로 출석이 매우 중요합니다
4. **준비물**: 노트북 필수 (실습용)

---

## 🎯 학기말 기대 성과물

### 개인
- 6개의 과제 포트폴리오
- 개인 GitHub Repository
- 물리 시뮬레이션 코드 모음

### 팀
- 최종 프로젝트 보고서
- 발표 자료
- 오픈소스 코드 (GitHub)

---

## 📚 추가 학습 자료 (선택)

### 온라인 강좌
- Andrew Ng's Machine Learning Course (Coursera)
- Fast.ai Practical Deep Learning
- Stanford CS231n (Convolutional Neural Networks)

### 유튜브 채널
- 3Blue1Brown (Neural Networks Series)
- Two Minute Papers
- Arxiv Insights

### 책
- *Neural Networks and Deep Learning* by Michael Nielsen (무료)
- *Dive into Deep Learning* by Zhang et al. (무료)
- *Computational Physics* by Giordano & Nakanishi

---

## 🔄 강의 계획 변경

- 본 계획서는 학습 진도에 따라 조정될 수 있습니다
- 중요한 변경사항은 최소 1주 전에 공지됩니다
- 학생 피드백을 반영하여 실습 내용을 조정할 수 있습니다

---

## 🎉 마치며

이 강의를 통해 여러분은 **21세기 물리학자에게 필수적인 AI 도구**를 
마스터하게 될 것입니다. 단순히 코드를 실행하는 것을 넘어,
**물리적 직관과 AI 기술을 결합하여 창의적인 문제 해결**을 
할 수 있기를 기대합니다.

함께 흥미로운 학기를 만들어갑시다! 🚀

---

**Last Updated**: 2025-11-20  
**Version**: 1.0  
**Instructor**: TBA