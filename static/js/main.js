document.addEventListener("DOMContentLoaded", () => {
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabPanels = document.querySelectorAll(".tab-panel");

  tabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const targetTabId = button.dataset.tab;

      tabButtons.forEach((btn) => btn.classList.remove("active"));
      button.classList.add("active");

      tabPanels.forEach((panel) => panel.classList.add("hidden"));
      document.getElementById(targetTabId).classList.remove("hidden");
    });
  });

  const fileInput = document.getElementById("fileInput");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const modelSelect = document.getElementById("modelSelect"); //  모델 선택

  // 결과 화면 요소들 
  const loading = document.getElementById("loading");
  const resultSection = document.getElementById("result-section");
  const resultContent = document.getElementById("result-content");
  const predText = document.getElementById("predText");
  
  // 비디오 요소
  const vidOriginal = document.getElementById("vidOriginal");
  const vidHeatmap = document.getElementById("vidHeatmap");

  // URL 관련 요소
  const urlAnalyzeBtn = document.getElementById("urlAnalyzeBtn");
  const clipToggleCheckbox = document.getElementById("clipToggleCheckbox");
  const timeInputsWrapper = document.getElementById("timeInputsWrapper");
  const urlInput = document.getElementById("urlInput");
  const startTimeInput = document.getElementById("startTimeInput");
  const endTimeInput = document.getElementById("endTimeInput");

  if (clipToggleCheckbox) {
    clipToggleCheckbox.addEventListener("change", () => {
      if (clipToggleCheckbox.checked) {
        timeInputsWrapper.classList.remove("hidden");
      } else {
        timeInputsWrapper.classList.add("hidden");
      }
    });
  }


  if (analyzeBtn) {
    analyzeBtn.addEventListener("click", async () => {
      if (fileInput.files.length === 0) {
        alert("분석할 동영상을 선택해주세요.");
        return;
      }

      // UI 초기화
      resultSection.classList.remove("hidden");
      loading.classList.remove("hidden");       // 로딩 표시
      if(resultContent) resultContent.classList.add("hidden"); // 결과 내용 숨김

      const formData = new FormData();
      formData.append("file", fileInput.files[0]);
      
  
      if (modelSelect) {
        formData.append("model_type", modelSelect.value);
      }

      try {
        const response = await fetch("/predict", {
          method: "POST",
          body: formData,
        });

        const data = await response.json();

        if (data.error) {
          throw new Error(data.error);
        }

    

        // 1. 텍스트 결과 표시
        if (data.is_fake) {
            predText.innerHTML = `결과: <span class="fake-text">FAKE (딥페이크 탐지됨)</span> <br> <span style="font-size:0.8em; color:#666;">확률: ${(data.prediction * 100).toFixed(2)}%</span>`;
        } else {
            predText.innerHTML = `결과: <span class="real-text">REAL (원본 영상)</span> <br> <span style="font-size:0.8em; color:#666;">확률: ${(100 - data.prediction * 100).toFixed(2)}%</span>`;
        }

        // 2. 비디오 소스 연결 
        const timestamp = new Date().getTime();
        vidOriginal.src = `${data.face_org_video}?t=${timestamp}`;
        vidHeatmap.src = `${data.face_map_video}?t=${timestamp}`;

        // 3. 비디오 동기화 설정
        setupVideoSync(vidOriginal, vidHeatmap);

        // 4. 그래프 그리기
        if (data.attention_graph) {
            drawChart(data.attention_graph);
        }

        // 로딩 끄고 결과 보여주기
        loading.classList.add("hidden");
        if(resultContent) resultContent.classList.remove("hidden");

      } catch (error) {
        console.error("Error:", error);
        alert("분석 중 오류가 발생했습니다: " + error.message);
        loading.classList.add("hidden");
      }
    });
  }

  // 4. URL 분석 로직 (간소화)
  if (urlAnalyzeBtn) {
    urlAnalyzeBtn.addEventListener("click", async () => {
      const url = urlInput.value.trim();
      if (!url) {
        alert("유튜브 URL을 입력해주세요.");
        return;
      }

      // 1. 서버로 전송할 데이터 구성
      const bodyData = { url: url };

      // 체크박스가 켜져 있을 때만 시간 값을 읽고 bodyData에 추가
      if (clipToggleCheckbox && clipToggleCheckbox.checked) {
        const startTime = startTimeInput.value.trim();
        const endTime = endTimeInput.value.trim();

        if (startTime && endTime) {
          bodyData.start_time = startTime;
          bodyData.end_time = endTime;
        } else {
          alert("시작 시간과 종료 시간을 모두 입력해주세요.");
          return;
        }
      }

      // 2. UI 초기화 (로딩 표시)
      resultSection.classList.remove("hidden");
      loading.classList.remove("hidden");
      if (resultContent) resultContent.classList.add("hidden");

      try {
        // 3. 서버 요청 (/predict_url 엔드포인트 사용)
        const response = await fetch("/predict_url", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(bodyData),
        });

        const data = await response.json();

        if (data.error) {
          throw new Error(data.error);
        }

        // 4. 결과 처리 및 화면 표시
        
        // (1) 텍스트 결과
        if (data.is_fake) {
            predText.innerHTML = `결과: <span class="fake-text">FAKE (딥페이크 탐지됨)</span> <br> <span style="font-size:0.8em; color:#666;">확률: ${(data.prediction * 100).toFixed(2)}%</span>`;
        } else {
            predText.innerHTML = `결과: <span class="real-text">REAL (원본 영상)</span> <br> <span style="font-size:0.8em; color:#666;">확률: ${(100 - data.prediction * 100).toFixed(2)}%</span>`;
        }

        // (2) 비디오 및 그래프 업데이트 (서버가 데이터를 줄 경우에만)
        // 주의: app.py의 /predict_url이 비디오 경로를 반환하도록 수정되어 있어야 영상이 뜹니다.
        if (data.face_org_video && data.face_map_video) {
            const timestamp = new Date().getTime();
            vidOriginal.src = `${data.face_org_video}?t=${timestamp}`;
            vidHeatmap.src = `${data.face_map_video}?t=${timestamp}`;
            setupVideoSync(vidOriginal, vidHeatmap);
        }

        if (data.attention_graph) {
            drawChart(data.attention_graph);
        }

        // 로딩 끄고 결과 보이기
        loading.classList.add("hidden");
        if (resultContent) resultContent.classList.remove("hidden");

      } catch (error) {
        console.error("Error:", error);
        alert("URL 분석 중 오류가 발생했습니다: " + error.message);
        loading.classList.add("hidden");
        resultSection.classList.add("hidden"); // 에러 시 결과창 다시 숨김
      }
    });
  }
});



// 비디오 동기화 함수
function setupVideoSync(v1, v2) {
    if (!v1 || !v2) return;

    // v1 제어 시 v2 동기화
    v1.onplay = () => v2.play();
    v1.onpause = () => v2.pause();
    v1.onseeking = () => v2.currentTime = v1.currentTime;
    v1.onseeked = () => v2.currentTime = v1.currentTime;
    
    // v2 제어 시 v1 동기화
    v2.onplay = () => v1.play();
    v2.onpause = () => v1.pause();
    v2.onseeking = () => v1.currentTime = v2.currentTime;
    v2.onseeked = () => v1.currentTime = v2.currentTime;
}

// Chart.js 그래프 그리기
let myChart = null; // 전역 변수로 차트 관리

function drawChart(scoreData) {
    const ctx = document.getElementById('attentionChart');
    if (!ctx) return;
    
    // 기존 차트 파괴 (중복 방지)
    if (myChart) {
        myChart.destroy();
    }

    const labels = scoreData.map((_, i) => `Frame ${i + 1}`);

    myChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Anomaly Score (Attention)',
                data: scoreData,
                borderColor: 'rgba(220, 53, 69, 1)', // 빨간색 선
                backgroundColor: 'rgba(220, 53, 69, 0.1)', // 아래 채우기
                borderWidth: 2,
                pointBackgroundColor: 'red',
                pointRadius: 4,
                fill: true,
                tension: 0.3 // 곡선
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: false, // [핵심] 0부터 시작하지 않고 데이터 범위에 맞춤
                    title: { display: true, text: 'Attention Intensity' },
                    grace: '5%' // [추천] 맨 위/아래에 5% 정도 여백을 줘서 보기 좋게 만듦
                }
            },
            interaction: {
                mode: 'index',
                intersect: false,
            },
        }
    });
}
