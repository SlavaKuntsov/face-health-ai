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
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false })
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

  const analyzeBlob = async (blob, baseSnapshotForPreview) => {
    setIsLoading(true); setError(null); setResult(null)
    setSnapshot(baseSnapshotForPreview || null)
    setAnnotatedSnapshot(null)

    try {
      const formData = new FormData()
      formData.append('image', blob, 'capture.jpg')

      const response = await fetch('/api/analyze', { method: 'POST', body: formData })
      if (!response.ok) {
        let detail = 'Ошибка анализа'
        try { const data = await response.json(); detail = data.detail || detail } catch {}
        throw new Error(detail)
      }
      const data = await response.json()
      setResult(data)

      let baseSnapshot = baseSnapshotForPreview
      if (!baseSnapshot) {
        baseSnapshot = await blobToDataURL(blob)
        setSnapshot(baseSnapshot)
      }
      await annotateSnapshot(baseSnapshot, data.faces || [])
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

  const labelColor = (label) => {
    if (label === 'болен') return 'rgba(220, 38, 38, 0.85)'      // red
    if (label === 'сомнительно') return 'rgba(234, 179, 8, 0.85)' // amber
    return 'rgba(22, 163, 74, 0.85)'                              // green
  }

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
        ctx.font = `${Math.max(14, Math.round(width / 32))}px sans-serif`
        ctx.textBaseline = 'middle'

        faces.forEach((face) => {
          const { x, y, width: fw, height: fh } = face
          const label = face?.assessment?.label || ''
          ctx.strokeStyle = label ? labelColor(label) : '#f87171'
          ctx.strokeRect(x, y, fw, fh)

          const text = label ? `${label}` : ''
          const labelHeight = Math.max(24, Math.round(width / 28))
          const labelWidth = Math.max(110, ctx.measureText(text).width + 24)
          const labelY = Math.max(labelHeight / 2, y - labelHeight / 2)
          ctx.fillStyle = label ? labelColor(label) : 'rgba(15,23,42,0.72)'
          ctx.fillRect(x, labelY - labelHeight / 2, labelWidth, labelHeight)
          ctx.fillStyle = '#f8fafc'
          ctx.fillText(text, x + 12, labelY)
        })

        setAnnotatedSnapshot(canvas.toDataURL('image/jpeg'))
        resolve()
      }
      image.src = baseSnapshot
    })

  const captureAndAnalyze = async () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!isStreaming || !video || !canvas) { setError('Сначала запустите камеру'); return }

    const width = video.videoWidth
    const height = video.videoHeight
    if (!width || !height) { setError('Камера ещё не готова'); return }

    canvas.width = width; canvas.height = height
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0, width, height)
    const baseSnapshot = canvas.toDataURL('image/jpeg')

    const blob = await new Promise((resolve, reject) => {
      canvas.toBlob(b => (b ? resolve(b) : reject(new Error('Не удалось получить изображение камеры'))), 'image/jpeg')
    })
    await analyzeBlob(blob, baseSnapshot)
  }

  const onUploadClick = () => fileInputRef.current?.click()
  const onFileChange = async (e) => {
    const file = e.target.files?.[0]; if (!file) return
    await analyzeBlob(file, null); e.target.value = ''
  }

  return React.createElement(
    'main',
    { className: 'container' },
    React.createElement('h1', null, 'FaceHealth: первичная оценка'),
    React.createElement('p', null, 'Снимите фото или загрузите файл — для каждого лица будет показана предварительная оценка.'),

    React.createElement(
      'section', { className: 'camera' },
      React.createElement('div', { className: 'video-frame' },
        React.createElement('video', { ref: videoRef, playsInline: true, muted: true, autoPlay: false })
      ),
      React.createElement('div', { className: 'controls' },
        React.createElement('button', { type: 'button', onClick: startCamera, disabled: isStreaming }, 'Запустить камеру'),
        React.createElement('button', { type: 'button', onClick: stopCamera, disabled: !isStreaming }, 'Остановить'),
        React.createElement('button', { type: 'button', onClick: captureAndAnalyze, disabled: !isStreaming || isLoading }, isLoading ? 'Анализ...' : 'Снимок и анализ'),
        React.createElement('button', { type: 'button', onClick: onUploadClick, disabled: isLoading }, 'Загрузить изображение'),
        React.createElement('input', { ref: fileInputRef, type: 'file', accept: 'image/*', style: { display: 'none' }, onChange: onFileChange })
      )
    ),

    error && React.createElement('div', { className: 'alert error' }, error),

    (annotatedSnapshot || snapshot) &&
      React.createElement('section', { className: 'preview' },
        React.createElement('h2', null, 'Последний снимок'),
        React.createElement('img', { src: annotatedSnapshot || snapshot, alt: 'Снимок/изображение' })
      ),

    result &&
      React.createElement('section', { className: 'results' },
        React.createElement('h2', null, 'Результаты'),
        React.createElement('p', null, `Лиц обнаружено: ${result.faces_count}`),

        // Глобальное качество
        result.quality && React.createElement('div', { className: 'quality' },
          React.createElement('h3', null, 'Качество кадра'),
          React.createElement('ul', null, [
            React.createElement('li', { key: 'q1' }, `blur_variance: ${result.quality.blur_variance}`),
            React.createElement('li', { key: 'q2' }, `brightness_mean: ${result.quality.brightness_mean}`),
            React.createElement('li', { key: 'q3' }, `face_area_ratio: ${result.quality.face_area_ratio}`),
            React.createElement('li', { key: 'q4' }, `detector: ${result.quality.detector}`),
            React.createElement('li', { key: 'q5' }, `ok: ${result.quality.ok}`),
            (result.quality.notes?.length ? React.createElement('li', { key: 'q6' }, `notes: ${result.quality.notes.join(', ')}`) : null)
          ])
        ),

        // По лицам
        result.faces && result.faces.length > 0 &&
          React.createElement('div', { className: 'faces' },
            React.createElement('h3', null, 'Оценка по каждому лицу'),
            React.createElement('ul', null,
              result.faces.map((f, i) =>
                React.createElement('li', { key: i },
                  React.createElement('div', null, `Оценка: ${f.assessment?.label || '—'} (score ${f.assessment?.score ?? 0})`),
                  React.createElement('div', null, `bbox: x=${f.x}, y=${f.y}, w=${f.width}, h=${f.height}`),
                  f.quality_local && React.createElement('div', null,
                    `quality: ok=${f.quality_local.ok}, blur=${f.quality_local.blur_variance}, bright=${f.quality_local.brightness_mean}`
                  ),
                  f.indicators && React.createElement('details', null,
                    React.createElement('summary', null, 'Индикаторы'),
                    React.createElement('ul', null, [
                      React.createElement('li', { key: 'r' }, `skin_redness: ${f.indicators.skin_redness}`),
                      React.createElement('li', { key: 'p' }, `skin_pallor: ${f.indicators.skin_pallor}`),
                      React.createElement('li', { key: 'y' }, `jaundice_like: ${f.indicators.jaundice_like}`),
                      React.createElement('li', { key: 'c' }, `cyanosis_like: ${f.indicators.cyanosis_like}`),
                      React.createElement('li', { key: 'e' }, `eye_sclera_yellowness: ${f.indicators.eye_sclera_yellowness}`),
                      React.createElement('li', { key: 'sp' }, `skin_pix: ${f.indicators.skin_pix}, sclera_pix: ${f.indicators.sclera_pix}`)
                    ])
                  )
                )
              )
            )
          ),

        (result.advice?.length > 0) && React.createElement('div', { className: 'advice' },
          React.createElement('h3', null, 'Рекомендации'),
          React.createElement('ul', null, result.advice.map((a, i) => React.createElement('li', { key: i }, a)))
        ),
        result.disclaimer && React.createElement('p', { className: 'disclaimer' }, result.disclaimer)
      ),

    React.createElement('canvas', { ref: canvasRef, style: { display: 'none' } })
  )
}

const root = ReactDOM.createRoot(document.getElementById('root'))
root.render(React.createElement(App))
