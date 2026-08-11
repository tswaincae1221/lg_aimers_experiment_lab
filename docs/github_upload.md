# GitHub에 처음 올리는 방법

대회 데이터 재배포 위험을 줄이기 위해 저장소는 우선 **Private**으로 만드는 것을
권장합니다.

## 1. GitHub에서 빈 저장소 만들기

1. GitHub 오른쪽 위 `+` → `New repository`
2. Repository name: `lg_aimers_experiment_lab`
3. `Private` 선택
4. `Add a README`, `.gitignore`, `license`는 선택하지 않음
5. `Create repository` 클릭

## 2. Windows PowerShell에서 올리기

이 ZIP의 압축을 풀고, `README.md`가 보이는 최상위 폴더에서 터미널을 엽니다.

```powershell
git init
git add .
git commit -m "Add feature and model experiment lab"
git branch -M main
git remote add origin https://github.com/tswaincae1221/lg_aimers_experiment_lab.git
git push -u origin main
```

GitHub 로그인이 필요하다는 창이 뜨면 브라우저 인증을 완료합니다.

## 3. 업로드 전 확인

```powershell
git status
git ls-files data
git ls-files results
```

정상 상태는 다음과 같습니다.

- `git status`: commit할 변경 없음
- `git ls-files data`: `data/README.md`만 표시
- `git ls-files results`: 아무것도 표시되지 않음

만약 원본 CSV나 결과 파일이 보이면 push하지 말고 `.gitignore`와 `git status`를 먼저
확인합니다.

## 4. 이후 수정 올리기

```powershell
git add .
git commit -m "Describe the experiment change"
git push
```

push 후 GitHub 저장소의 `Actions` 탭에서 `tests`가 초록색으로 통과하는지 확인합니다.
