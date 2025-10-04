const { useState, useRef, useEffect } = React;

function App() {
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [snapshot, setSnapshot] = useState(null);
  const [annotatedSnapshot, setAnnotatedSnapshot] = useState(null);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  const startCamera = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setError("Браузер не поддерживает доступ к камере");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) {
        throw new Error("Видеоэлемент не инициализирован");
      }
      video.srcObject = stream;
      await video.play();
      setIsStreaming(true);
      setError(null);
    } catch (err) {
      setError(`Не удалось получить доступ к камере: ${err.message}`);
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    const video = videoRef.current;
    if (video) {
      video.pause();
      video.srcObject = null;
    }
    setIsStreaming(false);
  };

  const captureAndAnalyze = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!isStreaming || !video || !canvas) {
      setError("Сначала запустите камеру и дождитесь отображения изображения");
      return;
    }

    const width = video.videoWidth;
    const height = video.videoHeight;

    if (!width || !height) {
      setError("Камера ещё не готова. Попробуйте снова через секунду.");
      return;
    }

    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    context.drawImage(video, 0, 0, width, height);
    const baseSnapshot = canvas.toDataURL("image/jpeg");

    setIsLoading(true);
    setError(null);
    setResult(null);
    setSnapshot(baseSnapshot);
    setAnnotatedSnapshot(null);

    try {
      const blob = await new Promise((resolve, reject) => {
        canvas.toBlob((blobValue) => {
          if (blobValue) {
            resolve(blobValue);
          } else {
            reject(new Error("Не удалось получить изображение камеры"));
          }
        }, "image/jpeg");
      });

      const formData = new FormData();
      formData.append("image", blob, "capture.jpg");

      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let detail = "Ошибка анализа";
        try {
          const data = await response.json();
          detail = data.detail || detail;
        } catch (_) {
          // ignore json parse issues
        }
        throw new Error(detail);
      }

      const data = await response.json();
      setResult(data);

      await new Promise((resolve) => {
        const image = new Image();
        image.onload = () => {
          context.drawImage(image, 0, 0, width, height);
          context.lineWidth = Math.max(2, Math.round(width / 240));
          context.strokeStyle = "#f87171";
          context.font = `${Math.max(14, Math.round(width / 32))}px sans-serif`;
          context.textBaseline = "middle";
          data.faces.forEach((face, index) => {
            context.strokeRect(face.x, face.y, face.width, face.height);
            const labelHeight = Math.max(24, Math.round(width / 28));
            const labelWidth = Math.max(80, Math.round(width / 10));
            const labelY = Math.max(labelHeight / 2, face.y - labelHeight / 2);
            context.fillStyle = "rgba(15, 23, 42, 0.72)";
            context.fillRect(face.x, labelY - labelHeight / 2, labelWidth, labelHeight);
            context.fillStyle = "#f8fafc";
            context.fillText(`#${index + 1}`, face.x + 12, labelY);
          });
          setAnnotatedSnapshot(canvas.toDataURL("image/jpeg"));
          resolve();
        };
        image.src = baseSnapshot;
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    React.createElement("main", { className: "container" },
      React.createElement("h1", null, "FaceIt: первичная оценка"),
      React.createElement("p", null, "Сделайте снимок с веб-камеры — система определит количество лиц в кадре."),
      React.createElement("section", { className: "camera" },
        React.createElement("div", { className: "video-frame" },
          React.createElement("video", {
            ref: videoRef,
            playsInline: true,
            muted: true,
            autoPlay: false,
          })
        ),
        React.createElement("div", { className: "controls" },
          React.createElement("button", {
            type: "button",
            onClick: startCamera,
            disabled: isStreaming,
          }, "Запустить камеру"),
          React.createElement("button", {
            type: "button",
            onClick: stopCamera,
            disabled: !isStreaming,
          }, "Остановить"),
          React.createElement("button", {
            type: "button",
            onClick: captureAndAnalyze,
            disabled: !isStreaming || isLoading,
          }, isLoading ? "Анализ..." : "Снимок и анализ"),
        )
      ),
      error && React.createElement("div", { className: "alert error" }, error),
      (annotatedSnapshot || snapshot) && React.createElement("section", { className: "preview" },
        React.createElement("h2", null, "Последний снимок"),
        React.createElement("img", {
          src: annotatedSnapshot || snapshot,
          alt: "Снимок с камеры",
        })
      ),
      result && React.createElement("section", { className: "results" },
        React.createElement("h2", null, "Результаты анализа"),
        React.createElement("p", null, `Количество обнаруженных лиц: ${result.faces_count}`),
        result.faces_count > 0 && React.createElement("details", null,
          React.createElement("summary", null, "Координаты лиц"),
          React.createElement("ul", null, result.faces.map((face, index) => (
            React.createElement("li", { key: index }, `Лицо ${index + 1}: x=${face.x}, y=${face.y}, ширина=${face.width}, высота=${face.height}`)
          )))
        )
      ),
      React.createElement("canvas", { ref: canvasRef, style: { display: "none" } })
    )
  );
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(React.createElement(App));
