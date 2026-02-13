import streamlit as st
import pandas as pd

# 페이지 설정
st.set_page_config(
    page_title="ResNet18 설명서",
    page_icon="🧠",
    layout="wide"
)

# CSS 스타일 적용 (가독성 향상)
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    h1 {
        color: #2e4053;
        font-family: 'Sans-serif', sans-serif;
    }
    h2 {
        color: #2874a6;
        border-bottom: 2px solid #2874a6;
        padding-bottom: 5px;
    }
    .highlight {
        background-color: #d4e6f1;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# 제목
st.title("🧠 ResNet-18 Architecture Deep Dive")
st.markdown("### 합성곱 신경망(CNN)의 혁명, Residual Learning의 시작")

# 1. 개요
st.header("1. ResNet이란? (Introduction)")
st.markdown("""
**ResNet(Residual Network)**은 2015년 ILSVRC 대회에서 압도적인 성적으로 우승하며 딥러닝의 역사를 바꾼 모델입니다.

가장 큰 특징은 **'잔차 학습(Residual Learning)'**을 통해 네트워크의 깊이(Depth)를 획기적으로 깊게 만들 수 있다는 점입니다.
- **문제점:** 네트워크가 깊어질수록 기울기 소실(Vanishing Gradient)이나 퇴화(Degradation) 문제가 발생하여 성능이 오히려 떨어지는 현상이 있었습니다.
- **해결책:** **Skip Connection(Shortcut)**을 도입하여, 입력 데이터를 레이어를 건너뛰어 출력에 더해주는 방식을 사용했습니다.
""")

# 2. 핵심 아이디어: Skip Connection
st.header("2. 핵심 아이디어: Skip Connection (Shortcut)")
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    일반적인 네트워크는 $H(x)$를 학습하려 하지만, ResNet은 **$F(x) = H(x) - x$**를 학습합니다.
    최종 출력은 $H(x) = F(x) + x$가 됩니다.
    
    이를 통해:
    1. 그라디언트가 손실 없이 하위 레이어로 전달됩니다.
    2. 레이어가 아무것도 학습하지 않는 경우(Identity mapping), 단순히 $F(x)=0$이 되면 입력이 그대로 출력되므로 성능 저하를 막을 수 있습니다.
    """)
with col2:
    # 텍스트로 다이어그램 표현
    st.code("""
    입력 (x)
      |
      +------------------+
      |                  |
      v                  |
    Conv Layer           |
      |                  |
    BatchNorm            |
      |                  |
    ReLU                 |
      |                  |
    Conv Layer           |
      v                  |
    [Add] <--------------+
      |
    ReLU
      |
    출력 (F(x) + x)
    """, language='text')

# 3. ResNet-18 구조 상세
st.header("3. ResNet-18 구조 상세 (Layer Details)")
st.markdown("ResNet-18은 총 **18개의 레이어**로 구성되어 있으며, 주로 이미지 분류(ImageNet)를 위해 설계되었습니다.")

# 테이블 데이터 생성
data = {
    'Layer Name': ['Conv1', 'MaxPool', 'Conv2_x', 'Conv3_x', 'Conv4_x', 'Conv5_x', 'AvgPool', 'FC'],
    'Output Size': ['112x112', '56x56', '56x56', '28x28', '14x14', '7x7', '1x1', '1000 (Classes)'],
    'Filter Size / Stride': ['7x7, 64, stride 2', '3x3, stride 2', '3x3, 64', '3x3, 128, stride 2', '3x3, 256, stride 2', '3x3, 512, stride 2', 'Adaptive Avg Pool', 'Fully Connected'],
    'Number of Blocks': ['-', '-', '2', '2', '2', '2', '-', '-'],
    'Total Layers': ['1 Conv', '-', '4 Convs', '4 Convs', '4 Convs', '4 Convs', '-', '1 FC']
}

df = pd.DataFrame(data)
st.table(df)

st.info("💡 **구조 포인트:** 각 `Conv_x` 블록은 2개의 BasicBlock을 가집니다. 각 BasicBlock은 2개의 Conv 레이어를 가지므로, 중간 레이어 수는 2 x 2 x 4 = 16개이며, 초기 Conv1과 마지막 FC를 합쳐 총 18개가 됩니다.")

# 4. BasicBlock 구조
st.header("4. BasicBlock 구조 (ResNet-18/34 전용)")
st.markdown("""
ResNet-18과 34는 **BasicBlock**이라는 작은 단위를 사용합니다. (ResNet-50 이상은 복잡한 BottleNeck 구조를 사용합니다.)

**BasicBlock 구성:**
1. 3x3 Conv
2. Batch Normalization
3. ReLU
4. 3x3 Conv
5. Batch Normalization
6. **Add (Input + Output)**
7. ReLU
""")

# 5. PyTorch 구현 예시
st.header("5. PyTorch 구현 예시 코드")
st.markdown("실제 PyTorch를 사용하여 ResNet-18을 불러오고 구조를 확인하는 코드입니다.")

code = """
import torch
import torch.nn as nn
import torchvision.models as models

# 1. 사전 훈련된 ResNet-18 모델 불러오기
model = models.resnet18(pretrained=True)

# 2. 모델 구조 출력
print(model)

# 3. 더미 데이터로 Forward Pass 테스트
dummy_input = torch.randn(1, 3, 224, 224) # Batch, Channel, Height, Width
output = model(dummy_input)
print(f"Output shape: {output.shape}") # torch.Size([1, 1000])

# 4. ResNet 커스터마이징 예시 (출력 클래스 변경)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10) # CIFAR-10 같은 10개 클래스용으로 변경
"""

st.code(code, language='python')

# 6. 요약
st.header("6. 요약 및 장단점")
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("✅ 장점")
    st.markdown("""
    - **깊은 네트워크 가능:** 기울기 소실 문제 해결로 수십, 수백 개의 레이어 사용 가능.
    - **높은 정확도:** 이미지 분류 벤치마크에서 강력한 성능.
    - **구현 용이성:** 구조가 간결하여 구현하고 변형하기 쉬움.
    """)

with col_b:
    st.subheader("⚠️ 단점/고려사항")
    st.markdown("""
    - **연산량:** ResNet-50 이상은 연산량이 많아 경량화가 필요할 수 있음.
    - **메모리 사용:** 깊이가 깊어 메모리 소모가 큼.
    - **과적합:** 데이터가 적을 경우 과적합 위험이 있음.
    """)

st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
