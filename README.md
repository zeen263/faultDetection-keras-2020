# faultDetection-keras-2020

신경망을 이용한 음향 측정 기반의 고장진단 시스템<br>
DOI : 10.7735/ksmte.2020.29.3.210<br><br>


모터를 사용하는 장치에 고장이 발생한 경우 문제가 생긴 부품에 따라 작동음이 상이하다는 점에서 착안하여, 장치의 정상 작동음과 고장이 발생한 경우의 작동음 데이터를 기반으로 딥 러닝 모델을 훈련시켜 장치의 고장 여부와 그 종류를 판단하는 시스템입니다. 냉각팬을 대상으로 *구동 정지, 날개 손상, 이물질 유입, 베어링 윤활제 부족, 축 마모* 등 발생 가능한 고장 상황을 정의하고 각 상황별로 작동음을 수집했습니다. 작동음을 수집할 때는 노이즈 캔슬링 기법을 사용해 장치 외부에서 발생하는 환경음의 영향을 소거했습니다. 일정 간격을 두고 수집한 작동음에 STFT(short-time Fourier transform)을 적용하여 스펙트로그램 형태로 나타내고, 작동음의 주파수 대역별 magnitude 벡터와 스펙트로그램 이미지를 사용하여 각각 MLP와 CNN 신경망을 학습시키고 성능을 평가했습니다.<br><br>

정상 구동 상태<br>
![img001](https://user-images.githubusercontent.com/42488309/123293741-34ce4c80-d54f-11eb-8ffa-d5b6cc1d857c.png)

블레이드 손상<br>
![img003](https://user-images.githubusercontent.com/42488309/123293781-3c8df100-d54f-11eb-8e60-8f0c07617c73.png)

회전축 마모<br>
![img002](https://user-images.githubusercontent.com/42488309/123293825-44e62c00-d54f-11eb-9c33-5bc8433d0d10.png)
<br><br>


FFT, 스펙트로그램 분석 관련된 부분은 adafruit 사에서 제공하는 코드를 이용했습니다.
https://learn.adafruit.com/fft-fun-with-fourier-transforms/software

<Br><br>


* 용량 문제로 데이터셋의 일부만 포함되어 있습니다.
