const { useState, useRef, useEffect } = React

function App() {
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [snapshot, setSnapshot] = useState(null)
  const [annotatedSnapshot, setAnnotatedSnapshot] = useState(null)

  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const fileInputRef = useRef(null)

  useEffect(() => () => stopCamera(), [])

  const startCamera = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setError('Браузер не поддерживает доступ к камере')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user' },
        audio: false
      })
      streamRef.current = stream
      const video = videoRef.current
      if (!video) throw new Error('Видеоэлемент не инициализирован')

      video.srcObject = stream
      await video.play()
      setIsStreaming(true)
      setError(null)
    } catch (err) {
      setError(`Не удалось получить доступ к камере: ${err.message}`)
    }
  }

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    const video = videoRef.current
    if (video) {
      video.pause()
      video.srcObject = null
    }
    setIsStreaming(false)
  }

  // Общая функция анализа двоичных данных изображения
  const analyzeBlob = async (blob, baseSnapshotForPreview) => {
    setIsLoading(true)
    setError(null)
    setResult(null)
    setSnapshot(baseSnapshotForPreview || null)
    setAnnotatedSnapshot(null)

    try {
      const formData = new FormData()
      formData.append('image', blob, 'capture.jpg')

      const response = await fetch('/api/analyze', { method: 'POST', body: formData })

      if (!response.ok) {
        let detail = 'Ошибка анализа'
        try {
          const data = await response.json()
          detail = data.detail || detail
        } catch (_) { /* ignore */ }
        throw new Error(detail)
      }

      const data = await response.json()
      setResult(data)

      // Если у нас ещё нет base64 — сделаем (для загруженного файла)
      let baseSnapshot = baseSnapshotForPreview
      if (!baseSnapshot) {
        baseSnapshot = await blobToDataURL(blob)
        setSnapshot(baseSnapshot)
      }

      // Нанесём рамки
      await annotateSnapshot(baseSnapshot, data.faces)
    } catch (err) {
      setError(err.message)
    } finally {
      setIsLoading(false)
    }
  }

  const blobToDataURL = (blob) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(reader.result)
      reader.onerror = reject
      reader.readAsDataURL(blob)
    })

  const annotateSnapshot = async (baseSnapshot, faces) =>
    new Promise(resolve => {
      const canvas = canvasRef.current
      const ctx = canvas.getContext('2d')
      const image = new Image()
      image.onload = () => {
        const width = image.naturalWidth
        const height = image.naturalHeight
        canvas.width = width
        canvas.height = height
        ctx.drawImage(image, 0, 0, width, height)
        ctx.lineWidth = Math.max(2, Math.round(width / 240))
        ctx.strokeStyle = '#f87171'
        ctx.font = `${Math.max(14, Math.round(width / 32))}px sans-serif`
        ctx.textBaseline = 'middle'

        faces.forEach((face, index) => {
          ctx.strokeRect(face.x, face.y, face.width, face.height)
          const labelHeight = Math.max(24, Math.round(width / 28))
          const labelWidth = Math.max(80, Math.round(width / 10))
          const labelY = Math.max(labelHeight / 2, face.y - labelHeight / 2)
          ctx.fillStyle = 'rgba(15, 23, 42, 0.72)'
          ctx.fillRect(face.x, labelY - labelHeight / 2, labelWidth, labelHeight)
          ctx.fillStyle = '#f8fafc'
          ctx.fillText(`#${index + 1}`, face.x + 12, labelY)
        })

        setAnnotatedSnapshot(canvas.toDataURL('image/jpeg'))
        resolve()
      }
      image.src = baseSnapshot
    })

  const captureAndAnalyze = async () => {
    const video = videoRef.current
    const canvas = canvasRef.current

    if (!isStreaming || !video || !canvas) {
      setError('Сначала запустите камеру и дождитесь отображения изображения')
      return
    }

    const width = video.videoWidth
    const height = video.videoHeight
    if (!width || !height) {
      setError('Камера ещё не готова. Попробуйте снова через секунду.')
      return
    }

    // Снимок
    canvas.width = width
    canvas.height = height
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0, width, height)
    const baseSnapshot = canvas.toDataURL('image/jpeg')

    // Получим blob и отправим
    const blob = await new Promise((resolve, reject) => {
      canvas.toBlob(b => (b ? resolve(b) : reject(new Error('Не удалось получить изображение камеры'))), 'image/jpeg')
    })

    await analyzeBlob(blob, baseSnapshot)
  }

  // Загрузка пользовательского изображения
  const onUploadClick = () => {
    fileInputRef.current?.click()
  }

  const onFileChange = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    await analyzeBlob(file, null) // base64 создадим внутри, чтобы показать превью
    // очистим input, чтобы повторно выбрать тот же файл можно было
    e.target.value = ''
  }

  return React.createElement(
    'main',
    { className: 'container' },
    React.createElement('h1', null, 'FaceHealth: первичная оценка'),
    React.createElement(
      'p',
      null,
      'Сделайте снимок с веб-камеры или загрузите файл — система определит количество лиц в кадре.'
    ),

    React.createElement(
      'section',
      { className: 'camera' },
      React.createElement(
        'div',
        { className: 'video-frame' },
        React.createElement('video', {
          ref: videoRef,
          playsInline: true,
          muted: true,
          autoPlay: false
        })
      ),
      React.createElement(
        'div',
        { className: 'controls' },
        React.createElement(
          'button',
          { type: 'button', onClick: startCamera, disabled: isStreaming },
          'Запустить камеру'
        ),
        React.createElement(
          'button',
          { type: 'button', onClick: stopCamera, disabled: !isStreaming },
          'Остановить'
        ),
        React.createElement(
          'button',
          { type: 'button', onClick: captureAndAnalyze, disabled: !isStreaming || isLoading },
          isLoading ? 'Анализ...' : 'Снимок и анализ'
        ),
        // Новое: кнопка загрузки
        React.createElement(
          'button',
          { type: 'button', onClick: onUploadClick, disabled: isLoading },
          'Загрузить изображение'
        ),
        React.createElement('input', {
          ref: fileInputRef,
          type: 'file',
          accept: 'image/*',
          style: { display: 'none' },
          onChange: onFileChange
        })
      )
    ),

    error && React.createElement('div', { className: 'alert error' }, error),

    (annotatedSnapshot || snapshot) &&
      React.createElement(
        'section',
        { className: 'preview' },
        React.createElement('h2', null, 'Последний снимок'),
        React.createElement('img', { src: annotatedSnapshot || snapshot, alt: 'Снимок/изображение' })
      ),

    result &&
      React.createElement(
        'section',
        { className: 'results' },
        React.createElement('h2', null, 'Результаты анализа'),
        React.createElement('p', null, `Количество обнаруженных лиц: ${result.faces_count}`),
        result.faces_count > 0 &&
          React.createElement(
            'details',
            null,
            React.createElement('summary', null, 'Координаты лиц'),
            React.createElement(
              'ul',
              null,
              result.faces.map((face, index) =>
                React.createElement(
                  'li',
                  { key: index },
                  `Лицо ${index + 1}: x=${face.x}, y=${face.y}, ширина=${face.width}, высота=${face.height}`
                )
              )
            )
          )
      ),

    React.createElement('canvas', { ref: canvasRef, style: { display: 'none' } })
  )
}

const root = ReactDOM.createRoot(document.getElementById('root'))
root.render(React.createElement(App))
