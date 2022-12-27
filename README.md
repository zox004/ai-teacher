# **Konkuk University 2022 Advanced Distributed System**

## **Project name : AI Teacher (Web Service for Classification Model with Kubernetes)**

## **Members**

[강신규](https://github.com/zox004) | [이승민](https://github.com/leeprac) | [조경혁](https://github.com/kyunghyukCHO)

## Introduction

교육에 있어서 학생들이 직접 참여하고 활동할 수 있게 하는 것은 매우 중요하다.교사의 설명을 일방적으로 듣기보다, 스스로 도구를 활용하고 응용하는 과정 속에서 수업 참여도나 효과가 더욱 높아지기 때문이다. 인공지능 교육 또한 중요성이 나날이 높아짐에 따라, 다양한 인공지능 교육 관련 학습도구들이 필요하다. AI Teacher는 사용자가 분류 인공지능 모델을 쉽고 빠르게 만들 수 있도록 제작된 웹 기반 플랫폼이며, 머신러닝에 관한 지식을 갖고 있지 않아도 충분히 모든 사람이 간편하게 활용할 수 있다. 머신러닝 코드를 작성하지 않고도 분류기 모델을 만들 수 있어 이를 자체 프로젝트나 사이트, 앱 등에서 사용할 수 있다. Google의 JS기반 Teachable Machine과 달리 Python기반 애플리케이션이기 때문에 state-of-the-art 모델로 개선하기 용이하다.

## Service Flow

1. 제작한 웹 서버를 도커 컨테이너로 배포 후 쿠버네티스를 이용해 효율적으로 리소스를 지속 관리한다.

2. 사용자는 AI Teacher에 해당하는 URL을 입력한다.
3. 정상적으로 웹 페이지 접근 후, 이미지 업로드, 모델 학습하기 등 웹 서비스를 사용할 수 있는 상태가 된다.
4. 사용자가 클래스명을 입력하고 이미지를 업로드 하는데 이때 이미지는 AWS S3에 저장되고, 모델 학습시키기를 클릭하고 학습이 완료된 모델은 .pt 파일로 DB에 저장된다
5. 쿠버네티스를 이용해 시스템 부하 시 오토스케일링을 통해 Scale out 후 리소스를 확장하고 대처한다. 또한 불필요한 리소스가 남아있을 경우 Scale in 후 리소스를 절약한다.
6. Classification 모델은 Flask 웹 서버와 DB에 연동하여 사용자가 모델 학습하기나 모델 추출하기를 실행했을 경우 DB에 저장된 데이터를 사용한다.

## Main Problem
- Local machine 리소스 한계
  - 딥러닝을 위해선 고성능의 GPU가 필요하며, local machine에 GPU가 없을 경우, CPU로 학습하기 때문에 장시간의 학습 시간 소요된다.
  
- 웹 서버와 AI 연동
  - 사용자가 해당 서비스를 이용하기 위해서 데이터를 전송해야 하고, pre-trained model은 사용자가 전송한 데이터로 추가 학습을 진행해야 한다.
  
- AWS EC2 Kubernetes 구축
  - 배포와 동시에 고성능의 GPU를 사용하기 위해 AWS EC2를 사용하기 때문에 EC2 Instance로 k8s cluster를 구축해야 한다.

## Solution
- Local machine 리소스 한계
  - AWS EC2의 cuda나 cudnn, torch, tensorflow 등 딥러닝을 위한 패키지가 미리 설치되어 있는 Deep Learning AMI (Amazon Machine Image)와 NVIDIA Tesla V100 GPU 지원하는 p3.2xlarge 인스턴스를 이용

- 웹 서버와 AI 연동
  - 사용자가 전송하는 데이터를 AWS S3에 저장한다. 모델은 S3에 저장된 데이터를 이용해서 학습을 진행하고 추가 학습이 진행된 모델은 .pt 파일로 DB에 저장한다.

- AWS EC2 Kubernetes 구축
  - AWS EKS를 이용하면 편리하게 k8s cluster를 구축할 수 있지만 비용이 많이 발생하며, Kubernetes를 이해하고 직접 설계하기 위해서 kops로 쿠버네티스 구축.

## Architecture

<img width=60% height=60% alt="image" src="https://user-images.githubusercontent.com/56228085/209655260-02bdd8e4-2ee4-41aa-a3be-ce9763228264.png">
