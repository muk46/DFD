// DOM이 모두 로드되었을 때 스크립트 실행
document.addEventListener("DOMContentLoaded", () => {
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabPanels = document.querySelectorAll(".tab-panel");

  tabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      // 0. 클릭한 버튼이 어떤 탭인지 확인 (e.g., "file" 또는 "url")
      const targetTabId = button.dataset.tab;

      // 1. 모든 버튼에서 'active' 클래스 제거
      tabButtons.forEach((btn) => btn.classList.remove("active"));

      // 2. 클릭한 버튼에만 'active' 클래스 추가
      button.classList.add("active");

      // 3. 모든 패널을 숨김 
      tabPanels.forEach((panel) => {
        panel.classList.add("hidden");
      });

      // 4. 타겟 패널을 보여줌 
      const targetPanel = document.getElementById(targetTabId);
      targetPanel.classList.remove("hidden");
    });
  });

  // 1. HTML에 있는 요소들을 가져옵니다.
  const fileInput = document.getElementById("fileInput");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const resultSection = document.getElementById("result-section");
  const resultBox = document.getElementById("result-box");
  const urlInput = document.getElementById("urlInput");
  const urlAnalyzeBtn = document.getElementById("urlAnalyzeBtn");
  const loading = document.getElementById("loading");
  const clipToggleCheckbox = document.getElementById("clipToggleCheckbox");
  const timeInputsWrapper = document.getElementById("timeInputsWrapper");
  const startTimeInput = document.getElementById("startTimeInput");
  const endTimeInput = document.getElementById("endTimeInput");

  if (clipToggleCheckbox) {
    clipToggleCheckbox.addEventListener("change", () => {
      timeInputsWrapper.classList.toggle("hidden", !clipToggleCheckbox.checked);
    });
  }

  // 1. 파일 분석 
  analyzeBtn.addEventListener("click", async () => {
    const files = Array.from(fileInput.files);
    if (files.length === 0) {
      alert("Please select one or more files to analyze.");
      return;
    }

    resultSection.classList.remove("hidden");
    loading.classList.remove("hidden"); // 로딩 보이기
    resultBox.innerHTML = ""; // 이전 결과 지우기
    resultBox.classList.add("hidden"); // 결과 상자 숨기기

    const analyzeFile = async (file) => {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 180000); 

      const formData = new FormData();
      formData.append("file", file);

      try {
        const response = await fetch("/predict", {
          method: "POST",
          body: formData,
          signal: controller.signal, 
        });

        clearTimeout(timeoutId);

        const data = await response.json();

        if (response.ok) {
          return formatResultDisplay(data, file.name);
        } else {
          return `<strong>${file.name}:</strong> <strong class="error">Error: ${data.error}</strong>`;
        }
      } catch (error) {
        clearTimeout(timeoutId); 

        if (error.name === "AbortError") {
          return `<strong>${file.name}:</strong> <strong class="error">Error: Analysis timed out (3 min).</strong>`;
        }
        console.error("Error analyzing file:", file.name, error);
        return `<strong>${file.name}:</strong> <strong class="error">An unexpected error occurred.</strong>`;
      }
    };

    const analysisPromises = files.map(analyzeFile);
    const results = await Promise.all(analysisPromises);

    loading.classList.add("hidden"); // 로딩 숨기기
    resultBox.innerHTML = results.join("<br>");
    resultBox.classList.remove("hidden"); // 결과 보이기

    fileInput.value = ""; // 파일 입력 초기화
  });

  // 2.  URL 분석 
  urlAnalyzeBtn.addEventListener("click", async () => {
    const url = urlInput.value.trim();
    if (!url) {
      alert("Please enter a YouTube URL.");
      return;
    }

    //  서버로 전송할 JSON 데이터 구성
    const bodyData = {
      url: url,
    };

    //  체크박스가 켜져 있을 때만 시간 값을 읽고 bodyData에 추가
    if (clipToggleCheckbox && clipToggleCheckbox.checked) {
      const startTime = startTimeInput.value.trim();
      const endTime = endTimeInput.value.trim();

      if (startTime && endTime) {
        bodyData.start_time = startTime;
        bodyData.end_time = endTime;
      } else {
        // 체크박스는 켰는데 시간을 입력 안 함
        alert("Please enter both Start and End time.");
        return;
      }
    }

    resultSection.classList.remove("hidden");
    loading.classList.remove("hidden"); // 로딩 보이기
    resultBox.innerHTML = ""; // 이전 결과 지우기
    resultBox.classList.add("hidden"); // 결과 상자 숨기기

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 180000); 

    try {
      const response = await fetch("/predict_url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(bodyData),
        signal: controller.signal, 
      });

      const data = await response.json();

      if (response.ok) {
        resultBox.innerHTML = formatResultDisplay(data, "YouTube URL");
      } else {
        resultBox.innerHTML = `<strong class="error">Error: ${data.error}</strong>`;
      }
    } catch (error) {
      if (error.name === "AbortError") {
        resultBox.innerHTML = `<strong class="error">Error: Analysis timed out (3 min).</strong>`;
      } else {
        console.error("Error analyzing URL:", url, error);
        resultBox.innerHTML = `<strong class="error">An unexpected network error occurred.</strong>`;
      }
    } finally {
      clearTimeout(timeoutId); 

      loading.classList.add("hidden"); // 로딩 숨기기
      resultBox.classList.remove("hidden"); // 결과 보이기
      urlInput.value = ""; // URL 입력창 비우기

      if (clipToggleCheckbox) {
        clipToggleCheckbox.checked = false;
        timeInputsWrapper.classList.add("hidden");
        startTimeInput.value = "";
        endTimeInput.value = "";
      }
    }
  });

  function formatResultDisplay(data, title = "") {
    let resultMessage = "";
    let titlePrefix = title ? `<strong>${title}:</strong> ` : "";
    
    let score = data.prediction !== undefined ? data.prediction : 0; 
    let percentage = 0;
    
    // 1. 텍스트 결과 생성
    if (data.is_fake) {
        percentage = (score * 100).toFixed(2);
        resultMessage = `${titlePrefix}<strong class="fake">FAKE</strong> <span style="font-size: 0.9em;">(${percentage}%)</span>`;
    } else {
        percentage = ((1 - score) * 100).toFixed(2);
        resultMessage = `${titlePrefix}<strong class="real">REAL</strong> <span style="font-size: 0.9em;">(${percentage}%)</span>`;
    }

    // 2.  XAI 이미지가 있으면 HTML에 추가
    if (data.xai_image) {
        resultMessage += `<br><div style="margin-top:15px; text-align:center;">
                            <p style="font-size:0.8em; color:#ccc;">▼ AI Attention Heatmap ▼</p>
                            <img src="${data.xai_image}" style="max-width: 200px; border-radius: 8px; border: 2px solid #555;">
                          </div>`;
    }

    return resultMessage;
  }
});