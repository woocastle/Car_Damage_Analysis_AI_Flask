# 차 파손 비용 견적 시뮬레이션

## 1. 구조 설명

```
/ - Flask Root Directory
/app - 시뮬레이션 App
  /__init__.py - App 실행 코드
  /source - 제공받은 원본 소스 (시뮬레이션 연산 코드)
    /__init__.py - App 연산 용 실행 코드
  /templates - 웹 페이지 템플릿 소스
    /main.html - 메인 화면 HTML 코드
```

## 2. 환경 세팅

*`venv` 활용한 구성 추천*

0. `venv` 사용 시

```bash
$ py -m venv venv
$ venv/Scripts/activate
```

1. `Flask` 패키지 설치

```bash
$ pip install flask
```

2. `requirements.txt` 설치

```bash
$ pip install -r app/source/requirements.txt
```

## 3. App 실행

`/` 루트 디렉터리에서 아래 실행

```bash
$ flask --app app run
 * Serving Flask app 'app'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
```

실행 후 http://127.0.0.1:5000/ URL로 접근하여 테스트
