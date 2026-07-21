import { AnimatePresence, motion } from 'motion/react'
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision'
import { useCallback, useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'

type Glyph = {
  ch: string
  x: number
  y: number
  width: number
  r: number
  rx: number
  ry: number
  fontSize: number
  ox: number
  oy: number
  vx: number
  vy: number
  angle: number
  angularVelocity: number
  mass: number
  held: boolean
}

type Drag = { index: number; dx: number; dy: number; source: 'hand' | 'pointer'; lastX: number; lastY: number; lastTime: number }
type HandPoint = { x: number; y: number }

type Params = {
  textScale: number
  spacing: number
  thresh: number
  wobble: number
  bump: number
  refract: number
  disperse: number
  fresnel: number
  frost: number
  irid: number
  shadow: number
}

const defaults: Params = {
  textScale: 1.67,
  spacing: 0,
  thresh: 0.47,
  wobble: 0,
  bump: 2.9,
  refract: 0.36,
  disperse: 0.09,
  fresnel: 3.2,
  frost: 0.006,
  irid: 0.7,
  shadow: 0.34,
}

const controls: Array<[keyof Params, string, number, number, number, number]> = [
  ['textScale', 'text size', 0.3, 2.5, 0.01, 2],
  ['spacing', 'spacing', 0, 0.35, 0.005, 3],
  ['thresh', 'gloop', 0.15, 0.85, 0.005, 3],
  ['wobble', 'wobble', 0, 0.03, 0.0002, 4],
  ['bump', 'bumpiness', 0.1, 6, 0.05, 2],
  ['refract', 'refraction', 0, 0.5, 0.002, 3],
  ['disperse', 'chromatic', 0, 0.2, 0.001, 3],
  ['fresnel', 'fresnel', 0, 6, 0.05, 2],
  ['frost', 'frost', 0, 0.1, 0.0005, 4],
  ['irid', 'iridescence', 0, 5, 0.05, 2],
  ['shadow', 'shadow', 0, 1, 0.01, 2],
]

const vertexShader = `#version 300 es
in vec2 a_position;
out vec2 v_uv;
void main() {
  v_uv = a_position * .5 + .5;
  gl_Position = vec4(a_position, 0., 1.);
}`

const fragmentShader = `#version 300 es
precision highp float;
uniform sampler2D u_background;
uniform sampler2D u_mask;
uniform sampler2D u_height;
uniform vec2 u_resolution;
uniform vec2 u_videoSize;
uniform float u_time;
uniform float u_camera;
uniform float u_thresh;
uniform float u_wobble;
uniform float u_bump;
uniform float u_refract;
uniform float u_disperse;
uniform float u_fresnel;
uniform float u_frost;
uniform float u_irid;
uniform float u_shadow;
uniform float u_fire;
in vec2 v_uv;
out vec4 outColor;

float rand(vec2 p) { return fract(sin(dot(p, vec2(127.1,311.7))) * 43758.5453); }
vec3 sky(vec2 uv) {
  float t = u_time * .07;
  vec2 p = uv - .5;
  float glow = exp(-5.2 * length(p - vec2(-.18 + sin(t) * .12, .12)));
  float glow2 = exp(-8. * length(p - vec2(.34, -.15 + cos(t*.8)*.08)));
  vec3 top = vec3(.055, .12, .22);
  vec3 bot = vec3(.7, .34, .3);
  vec3 col = mix(bot, top, smoothstep(0., .95, uv.y));
  col += glow * vec3(.8,.48,.25) + glow2 * vec3(.2,.5,.7);
  col += .025 * sin(vec3(1.,1.4,1.8) * (uv.x * 13. + uv.y * 8. + t));
  return col;
}
vec3 bg(vec2 uv) {
  if (u_camera < .5) return sky(uv);
  float screenAspect = u_resolution.x / u_resolution.y;
  float videoAspect = u_videoSize.x / max(1., u_videoSize.y);
  vec2 cameraUv = uv;
  if (videoAspect > screenAspect) cameraUv.x = .5 + (cameraUv.x - .5) * screenAspect / videoAspect;
  else cameraUv.y = .5 + (cameraUv.y - .5) * videoAspect / screenAspect;
  cameraUv.x = 1. - cameraUv.x;
  return texture(u_background, clamp(cameraUv, .001, .999)).rgb;
}
float maskAt(vec2 uv) {
  float wob = u_wobble * sin(uv.y * 31. + u_time * 2.1) * sin(uv.x * 19. - u_time);
  return texture(u_mask, uv + vec2(wob, wob * .45)).r;
}
float edgeBand(float value, float threshold) {
  return smoothstep(threshold - .04, threshold - .03, value)
    * (1. - smoothstep(threshold - .03, threshold - .01, value));
}
void main() {
  vec2 uv = v_uv;
  vec2 px = 1. / u_resolution;
  float m = maskAt(uv);
  float radius = 1.2 + u_bump * .34;
  float height = texture(u_height, uv).r;
  float hL = texture(u_height, uv - vec2(px.x * radius, 0.)).r;
  float hR = texture(u_height, uv + vec2(px.x * radius, 0.)).r;
  float hD = texture(u_height, uv - vec2(0., px.y * radius)).r;
  float hU = texture(u_height, uv + vec2(0., px.y * radius)).r;
  float gloop = clamp((u_thresh - .15) / .7, 0., 1.);
  float fusionThreshold = mix(.52, .24, gloop);
  float gate = smoothstep(fusionThreshold - .055, fusionThreshold + .035, max(m * .9, height));
  vec2 slope = vec2(hR - hL, hU - hD);
  float slopeLength = length(slope);
  vec2 normal = slopeLength > .00001 ? normalize(slope) : vec2(0.);
  float bevel = smoothstep(.012, .072, slopeLength);
  float inner = smoothstep(fusionThreshold + .02, .88, height);
  vec2 surfaceWarp = vec2(sin(uv.y * 19. + u_time * .35), cos(uv.x * 17. - u_time * .3)) * .002 * inner;
  vec2 bend = normal * u_refract * mix(.082, .03, inner) + surfaceWarp * u_refract;
  float dispersion = u_disperse * .012;
  vec3 refracted;
  refracted.r = bg(uv + bend + normal * dispersion).r;
  refracted.g = bg(uv + bend).g;
  refracted.b = bg(uv + bend - normal * dispersion).b;
  float noise = rand(floor(uv * u_resolution * .45) + floor(u_time * 8.));
  refracted = mix(refracted, vec3(dot(refracted, vec3(.299,.587,.114))), u_frost * 1.45);
  refracted += (noise - .5) * u_frost * .24;
  float rim = pow(clamp(bevel * 1.3, 0., 1.), max(.9, 4.8 - u_fresnel * .48));
  vec3 rainbow = .5 + .5 * cos(6.283 * (vec3(0.,.33,.67) + atan(normal.y, normal.x)/6.283 + u_time*.025));
  vec3 glass = refracted;
  glass = mix(glass, vec3(1.), rim * .3);
  glass += rainbow * rim * u_irid * .012;
  glass += vec3(.075) * rim * smoothstep(fusionThreshold, .8, height);
  float specular = slopeLength > .00001 ? pow(max(0., dot(normalize(vec3(normal, .72)), normalize(vec3(-.35,.55,.76)))), 34.) : 0.;
  float caustic = slopeLength > .00001 ? pow(max(0., dot(normal, normalize(vec2(.55,.82)))), 5.) * inner : 0.;
  glass += specular * rim * .24 + caustic * .045;
  if (u_fire > .5) {
    vec3 fire = mix(vec3(.55,.015,.005), vec3(1.,.78,.08), clamp(m + uv.y*.5, 0., 1.));
    glass = mix(fire, vec3(1.,.9,.35), rim);
  }
  float shadowMask = texture(u_height, uv + vec2(-px.x*20., px.y*26.)).r;
  float shadow = smoothstep(.09,.52,shadowMask) * (1. - gate) * u_shadow * .52;
  vec3 base = bg(uv);
  base *= 1. - shadow;
  vec3 col = mix(base, glass, gate);
  float fringeOffset = 1.6 + u_disperse * 10.;
  float fringeR = edgeBand(texture(u_height, uv + normal * px * fringeOffset).r, fusionThreshold);
  float fringeG = edgeBand(height, fusionThreshold);
  float fringeB = edgeBand(texture(u_height, uv - normal * px * fringeOffset).r, fusionThreshold);
  vec3 spectralLine = fringeR * vec3(0., .82, 1.)
    + fringeG * vec3(1., .04, .72)
    + fringeB * vec3(1., .86, 0.);
  col += spectralLine * (1. - gate * .72) * .46;
  outColor = vec4(col, 1.);
}`

const blurFragmentShader = `#version 300 es
precision highp float;
uniform sampler2D u_source;
uniform vec2 u_texel;
uniform vec2 u_direction;
uniform float u_radius;
in vec2 v_uv;
out vec4 outColor;
void main() {
  vec2 stepUv = u_texel * u_direction * u_radius;
  float value = texture(u_source, v_uv).r * .227027;
  value += texture(u_source, v_uv + stepUv * 1.384615).r * .316216;
  value += texture(u_source, v_uv - stepUv * 1.384615).r * .316216;
  value += texture(u_source, v_uv + stepUv * 3.230769).r * .070270;
  value += texture(u_source, v_uv - stepUv * 3.230769).r * .070270;
  outColor = vec4(value, value, value, 1.);
}`

function createShader(gl: WebGL2RenderingContext, type: number, source: string) {
  const shader = gl.createShader(type)!
  gl.shaderSource(shader, source)
  gl.compileShader(shader)
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader) || 'Shader compile failed')
  return shader
}

function createProgram(gl: WebGL2RenderingContext, fragment: string) {
  const program = gl.createProgram()!
  gl.attachShader(program, createShader(gl, gl.VERTEX_SHADER, vertexShader))
  gl.attachShader(program, createShader(gl, gl.FRAGMENT_SHADER, fragment))
  gl.linkProgram(program)
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) || 'Program link failed')
  return program
}

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const paramsRef = useRef(defaults)
  const textRef = useRef('LIQUID GLASS')
  const cameraReadyRef = useRef(false)
  const cameraChoiceRef = useRef('camera')
  const styleRef = useRef('liquid glass')
  const glyphsRef = useRef<Glyph[]>([])
  const dragRef = useRef<Drag | null>(null)
  const resetEpochRef = useRef(0)
  const [params, setParams] = useState(defaults)
  const [text, setText] = useState('LIQUID GLASS')
  const [collapsed, setCollapsed] = useState(false)
  const [hidden, setHidden] = useState(false)
  const [fps, setFps] = useState(60)
  const [cameraNotice, setCameraNotice] = useState('')
  const [cameraChoice, setCameraChoice] = useState('camera')
  const [cameraName, setCameraName] = useState('camera')
  const [style, setStyle] = useState('liquid glass')
  const [handStatus, setHandStatus] = useState<'LOADING' | 'READY' | 'NO HAND'>('LOADING')
  const [handPoints, setHandPoints] = useState<{ thumb: HandPoint; index: HandPoint; pinch: boolean } | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => { paramsRef.current = params }, [params])
  useEffect(() => { textRef.current = text }, [text])
  useEffect(() => { styleRef.current = style }, [style])
  useEffect(() => { cameraChoiceRef.current = cameraChoice }, [cameraChoice])

  const reset = useCallback(() => {
    setParams(defaults)
    setText('LIQUID GLASS')
    resetEpochRef.current++
    for (const glyph of glyphsRef.current) {
      glyph.ox = 0; glyph.oy = 0; glyph.vx = 0; glyph.vy = 0; glyph.angle = 0; glyph.angularVelocity = 0; glyph.held = false
    }
    dragRef.current = null
  }, [])

  const releaseDrag = useCallback((source?: Drag['source']) => {
    const drag = dragRef.current
    if (!drag || (source && drag.source !== source)) return
    const glyph = glyphsRef.current[drag.index]
    if (glyph) glyph.held = false
    dragRef.current = null
  }, [])

  const pickGlyph = useCallback((x: number, y: number, source: Drag['source']) => {
    let best = -1
    let bestDistance = Infinity
    glyphsRef.current.forEach((glyph, index) => {
      if (!glyph.ch.trim()) return
      const dx = x - glyph.x - glyph.ox
      const dy = y - glyph.y - glyph.oy
      const cosine = Math.cos(glyph.angle), sine = Math.sin(glyph.angle)
      const localX = cosine * dx + sine * dy
      const localY = -sine * dx + cosine * dy
      const hit = (localX / Math.max(1, glyph.width * .58)) ** 2 + (localY / Math.max(1, glyph.fontSize * .43)) ** 2
      const distance = Math.hypot(dx, dy)
      if (hit <= 1.25 && distance < bestDistance) { best = index; bestDistance = distance }
    })
    if (best < 0) return false
    releaseDrag()
    const glyph = glyphsRef.current[best]
    glyph.held = true
    glyph.vx = 0
    glyph.vy = 0
    dragRef.current = { index: best, dx: glyph.x + glyph.ox - x, dy: glyph.y + glyph.oy - y, source, lastX: x, lastY: y, lastTime: performance.now() }
    return true
  }, [releaseDrag])

  const moveDrag = useCallback((x: number, y: number, source: Drag['source']) => {
    const drag = dragRef.current
    if (!drag || drag.source !== source) return
    const glyph = glyphsRef.current[drag.index]
    if (!glyph) return
    const now = performance.now()
    const dt = Math.max(.008, Math.min(.08, (now - drag.lastTime) / 1000))
    const maxThrowSpeed = 2400 * Math.min(devicePixelRatio, 1.25)
    const sampleVx = Math.max(-maxThrowSpeed, Math.min(maxThrowSpeed, (x - drag.lastX) / dt))
    const sampleVy = Math.max(-maxThrowSpeed, Math.min(maxThrowSpeed, (y - drag.lastY) / dt))
    glyph.vx += (sampleVx - glyph.vx) * .42
    glyph.vy += (sampleVy - glyph.vy) * .42
    glyph.angularVelocity += ((sampleVx / Math.max(1, glyph.r)) * .28 - glyph.angularVelocity) * .18
    glyph.angularVelocity = Math.max(-8, Math.min(8, glyph.angularVelocity))
    glyph.ox = x + drag.dx - glyph.x
    glyph.oy = y + drag.dy - glyph.y
    drag.lastX = x
    drag.lastY = y
    drag.lastTime = now
  }, [])

  const pointerPosition = useCallback((event: ReactPointerEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect()
    return { x: (event.clientX - rect.left) * event.currentTarget.width / rect.width, y: (event.clientY - rect.top) * event.currentTarget.height / rect.height }
  }, [])

  const onPointerDown = useCallback((event: ReactPointerEvent<HTMLCanvasElement>) => {
    const point = pointerPosition(event)
    if (pickGlyph(point.x, point.y, 'pointer')) event.currentTarget.setPointerCapture(event.pointerId)
  }, [pickGlyph, pointerPosition])

  const onPointerMove = useCallback((event: ReactPointerEvent<HTMLCanvasElement>) => {
    const point = pointerPosition(event)
    moveDrag(point.x, point.y, 'pointer')
  }, [moveDrag, pointerPosition])

  useEffect(() => {
    let stream: MediaStream | null = null
    let cancelled = false
    async function startCamera() {
      if (cameraChoice === 'none') {
        cameraReadyRef.current = false
        setCameraNotice('')
        return
      }
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }, audio: false })
        if (cancelled) { stream.getTracks().forEach((track) => track.stop()); return }
        const video = videoRef.current!
        video.srcObject = stream
        await video.play()
        const track = stream.getVideoTracks()[0]
        setCameraName(track.getSettings().deviceId ? (track.label || 'camera') : 'camera')
        cameraReadyRef.current = true
        setCameraNotice('')
      } catch {
        cameraReadyRef.current = false
        setCameraNotice('CAMERA UNAVAILABLE · USING PROCEDURAL SKY')
      }
    }
    startCamera()
    return () => { cancelled = true; stream?.getTracks().forEach((track) => track.stop()) }
  }, [cameraChoice])

  useEffect(() => {
    let stopped = false
    let landmarker: HandLandmarker | null = null
    let raf = 0
    let lastDetection = 0
    let pinching = false
    let smoothThumb: HandPoint | null = null
    let smoothIndex: HandPoint | null = null
    const inferenceCanvas = document.createElement('canvas')
    const inferenceContext = inferenceCanvas.getContext('2d', { alpha: false })!

    const mapToScreen = (point: { x: number; y: number }, video: HTMLVideoElement, canvas: HTMLCanvasElement) => {
      const screenAspect = canvas.width / canvas.height
      const videoAspect = video.videoWidth / Math.max(1, video.videoHeight)
      const scaleX = videoAspect > screenAspect ? screenAspect / videoAspect : 1
      const scaleY = videoAspect > screenAspect ? 1 : videoAspect / screenAspect
      return { x: .5 + ((1 - point.x) - .5) / scaleX, y: .5 + (point.y - .5) / scaleY }
    }

    const tick = (now: number) => {
      if (stopped) return
      const video = videoRef.current
      const canvas = canvasRef.current
      if (landmarker && video && canvas && cameraReadyRef.current && cameraChoiceRef.current === 'camera' && video.readyState >= 2 && now - lastDetection >= 50) {
        lastDetection = now
        const videoWidth = Math.max(1, video.videoWidth)
        const videoHeight = Math.max(1, video.videoHeight)
        const inferenceScale = Math.min(1, 640 / Math.max(videoWidth, videoHeight))
        const inferenceWidth = Math.max(1, Math.round(videoWidth * inferenceScale))
        const inferenceHeight = Math.max(1, Math.round(videoHeight * inferenceScale))
        if (inferenceCanvas.width !== inferenceWidth || inferenceCanvas.height !== inferenceHeight) {
          inferenceCanvas.width = inferenceWidth
          inferenceCanvas.height = inferenceHeight
        }
        inferenceContext.drawImage(video, 0, 0, inferenceWidth, inferenceHeight)
        const hand = landmarker.detectForVideo(inferenceCanvas, now).landmarks[0]
        if (hand) {
          const thumbRaw = mapToScreen(hand[4], video, canvas)
          const indexRaw = mapToScreen(hand[8], video, canvas)
          const alpha = .35
          smoothThumb = smoothThumb ? { x: smoothThumb.x + (thumbRaw.x - smoothThumb.x) * alpha, y: smoothThumb.y + (thumbRaw.y - smoothThumb.y) * alpha } : thumbRaw
          smoothIndex = smoothIndex ? { x: smoothIndex.x + (indexRaw.x - smoothIndex.x) * alpha, y: smoothIndex.y + (indexRaw.y - smoothIndex.y) * alpha } : indexRaw
          const pinchDistance = Math.hypot(hand[4].x - hand[8].x, hand[4].y - hand[8].y)
          const palmScale = Math.max(.001, Math.hypot(hand[0].x - hand[5].x, hand[0].y - hand[5].y))
          const ratio = pinchDistance / palmScale
          const nextPinching = pinching ? ratio < .48 : ratio < .32
          const midpoint = { x: (smoothThumb.x + smoothIndex.x) * .5, y: (smoothThumb.y + smoothIndex.y) * .5 }
          const px = midpoint.x * canvas.width
          const py = midpoint.y * canvas.height
          if (nextPinching && !pinching) pickGlyph(px, py, 'hand')
          if (nextPinching) moveDrag(px, py, 'hand')
          if (!nextPinching && pinching) releaseDrag('hand')
          pinching = nextPinching
          setHandPoints({ thumb: smoothThumb, index: smoothIndex, pinch: pinching })
          setHandStatus('READY')
        } else {
          if (pinching) releaseDrag('hand')
          pinching = false
          smoothThumb = null
          smoothIndex = null
          setHandPoints(null)
          setHandStatus('NO HAND')
        }
      }
      raf = requestAnimationFrame(tick)
    }

    async function initialize() {
      try {
        const vision = await FilesetResolver.forVisionTasks('/mediapipe/wasm')
        if (stopped) return
        landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: '/mediapipe/models/hand_landmarker.task', delegate: 'CPU' },
          runningMode: 'VIDEO', numHands: 1,
          minHandDetectionConfidence: .55, minHandPresenceConfidence: .5, minTrackingConfidence: .5,
        })
        if (stopped) { landmarker.close(); return }
        setHandStatus(cameraChoiceRef.current === 'camera' ? 'READY' : 'NO HAND')
        raf = requestAnimationFrame(tick)
      } catch {
        if (!stopped) setHandStatus('NO HAND')
      }
    }
    initialize()
    return () => {
      stopped = true
      cancelAnimationFrame(raf)
      releaseDrag('hand')
      landmarker?.close()
    }
  }, [moveDrag, pickGlyph, releaseDrag])

  useEffect(() => {
    const canvas = canvasRef.current!
    const gl = canvas.getContext('webgl2', { alpha: false, antialias: true })
    if (!gl) { setCameraNotice('WebGL2 unavailable'); return }
    const program = createProgram(gl, fragmentShader)
    const blurProgram = createProgram(gl, blurFragmentShader)
    const quad = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, quad)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 3,-1, -1,3]), gl.STATIC_DRAW)

    const bindProgram = (target: WebGLProgram) => {
      gl.useProgram(target)
      gl.bindBuffer(gl.ARRAY_BUFFER, quad)
      const position = gl.getAttribLocation(target, 'a_position')
      gl.enableVertexAttribArray(position)
      gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0)
    }
    const configureTexture = (texture: WebGLTexture) => {
      gl.bindTexture(gl.TEXTURE_2D, texture)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    }

    const bgTexture = gl.createTexture()!
    gl.activeTexture(gl.TEXTURE0)
    configureTexture(bgTexture)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([20,35,55,255]))

    const maskCanvas = document.createElement('canvas')
    const maskCtx = maskCanvas.getContext('2d')!
    const maskTexture = gl.createTexture()!
    gl.activeTexture(gl.TEXTURE1)
    configureTexture(maskTexture)
    const blurA = gl.createTexture()!
    const blurB = gl.createTexture()!
    const framebuffer = gl.createFramebuffer()!
    let halfWidth = 1
    let halfHeight = 1
    let layoutKey = ''
    let raf = 0
    let frames = 0
    let fpsAt = performance.now()
    let physicsAt = performance.now()
    let physicsAccumulator = 0

    function resize() {
      const dpr = Math.min(devicePixelRatio, 1.25)
      const w = Math.floor(innerWidth * dpr)
      const h = Math.floor(innerHeight * dpr)
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = maskCanvas.width = w
        canvas.height = maskCanvas.height = h
        halfWidth = Math.max(1, Math.floor(w / 2))
        halfHeight = Math.max(1, Math.floor(h / 2))
        for (const texture of [blurA, blurB]) {
          gl!.bindTexture(gl!.TEXTURE_2D, texture)
          configureTexture(texture)
          gl!.texImage2D(gl!.TEXTURE_2D, 0, gl!.RGBA8, halfWidth, halfHeight, 0, gl!.RGBA, gl!.UNSIGNED_BYTE, null)
        }
        layoutKey = ''
      }
    }

    function rebuildLayout() {
      const p = paramsRef.current
      const w = maskCanvas.width, h = maskCanvas.height
      const maxWidth = w * .88
      let size = Math.min(w * .142, h * .24) * p.textScale
      maskCtx.font = `800 ${size}px "Baloo 2", "Arial Rounded MT Bold", sans-serif`
      const words = textRef.current.trim().toUpperCase().split(/\s+/).filter(Boolean)
      const trackedWidth = (value: string, tracking: number) => {
        let width = 0
        for (const char of value) width += maskCtx.measureText(char).width
        return width + Math.max(0, value.length - 1) * tracking
      }
      const makeLines = (tracking: number) => {
        const result: string[] = []
        let current = ''
        for (const word of words) {
          const candidate = current ? `${current} ${word}` : word
          if (current && trackedWidth(candidate, tracking) > maxWidth) {
            result.push(current)
            current = word
          } else current = candidate
          while (trackedWidth(current, tracking) > maxWidth && current.length > 1) {
            let splitAt = current.length - 1
            while (splitAt > 1 && trackedWidth(current.slice(0, splitAt), tracking) > maxWidth) splitAt--
            result.push(current.slice(0, splitAt))
            current = current.slice(splitAt).trimStart()
          }
        }
        if (current) result.push(current)
        return result.length ? result : [' ']
      }
      let lines: string[] = []
      let tracking = p.spacing * size
      lines = makeLines(tracking)
      while (lines.length > 4 && size > 34) {
        size *= .94
        maskCtx.font = `800 ${size}px "Baloo 2", "Arial Rounded MT Bold", sans-serif`
        tracking = p.spacing * size
        lines = makeLines(tracking)
      }
      const lineHeight = size
      const startY = h * .53 - ((lines.length - 1) * lineHeight) / 2
      const glyphs: Glyph[] = []
      lines.slice(0,4).forEach((value, index) => {
        let x = (w - trackedWidth(value, tracking)) / 2
        for (const char of value) {
          const width = maskCtx.measureText(char).width
          const radiusX = Math.min(size * .38, Math.max(size * .13, width * .43))
          const radiusY = size * .36
          glyphs.push({
            ch: char, x: x + width / 2, y: startY + index * lineHeight, width, r: Math.sqrt(radiusX * radiusY), rx: radiusX, ry: radiusY, fontSize: size,
            ox: 0, oy: 0, vx: 0, vy: 0, angle: 0, angularVelocity: 0,
            mass: Math.max(.65, width / Math.max(1, size)), held: false,
          })
          x += width + tracking
        }
      })
      glyphsRef.current = glyphs
    }

    function drawMask() {
      const p = paramsRef.current
      const nextKey = `${maskCanvas.width}:${maskCanvas.height}:${textRef.current}:${p.textScale}:${p.spacing}:${resetEpochRef.current}`
      if (nextKey !== layoutKey) { layoutKey = nextKey; rebuildLayout() }
      const w = maskCanvas.width, h = maskCanvas.height
      maskCtx.clearRect(0,0,w,h)
      maskCtx.fillStyle = 'white'
      maskCtx.strokeStyle = 'white'
      maskCtx.lineCap = 'round'
      maskCtx.lineJoin = 'round'
      const visible = glyphsRef.current.filter((glyph) => glyph.ch.trim())
      for (let i = 0; i < visible.length; i++) {
        for (let j = i + 1; j < visible.length; j++) {
          const a = visible[i], b = visible[j]
          const ax = a.x + a.ox, ay = a.y + a.oy, bx = b.x + b.ox, by = b.y + b.oy
          const gapX = Math.max(0, Math.abs(ax - bx) - (a.width + b.width) * .5)
          const gapY = Math.max(0, Math.abs(ay - by) - (a.r + b.r) * .72)
          if (Math.hypot(gapX, gapY) <= 5 * Math.min(devicePixelRatio, 1.25)) {
            maskCtx.lineWidth = Math.min(a.r, b.r) * .16
            maskCtx.beginPath(); maskCtx.moveTo(ax, ay); maskCtx.lineTo(bx, by); maskCtx.stroke()
          }
        }
      }
      maskCtx.textAlign = 'center'
      maskCtx.textBaseline = 'middle'
      for (const glyph of glyphsRef.current) {
        if (!glyph.ch.trim()) continue
        maskCtx.font = `800 ${glyph.fontSize}px "Baloo 2", "Arial Rounded MT Bold", sans-serif`
        maskCtx.save()
        maskCtx.translate(glyph.x + glyph.ox, glyph.y + glyph.oy)
        maskCtx.rotate(glyph.angle)
        maskCtx.fillText(glyph.ch, 0, 0)
        maskCtx.restore()
      }
      gl!.activeTexture(gl!.TEXTURE1)
      gl!.bindTexture(gl!.TEXTURE_2D, maskTexture)
      gl!.pixelStorei(gl!.UNPACK_FLIP_Y_WEBGL, true)
      gl!.texImage2D(gl!.TEXTURE_2D, 0, gl!.R8, gl!.RED, gl!.UNSIGNED_BYTE, maskCanvas)
      gl!.pixelStorei(gl!.UNPACK_FLIP_Y_WEBGL, false)
    }

    function stepPhysics(dt: number) {
      const glyphs = glyphsRef.current.filter((glyph) => glyph.ch.trim())
      const gravity = canvas.height * 1.35
      const linearDamping = Math.pow(.994, dt * 60)
      const angularDamping = Math.pow(.988, dt * 60)
      const velocityLimit = 3200 * Math.min(devicePixelRatio, 1.25)
      const smoothImpact = (speed: number, low: number, high: number, minimum: number, maximum: number) => {
        const linear = Math.max(0, Math.min(1, (speed - low) / (high - low)))
        const eased = linear * linear * (3 - 2 * linear)
        return minimum + (maximum - minimum) * eased
      }
      for (const glyph of glyphs) {
        if (!glyph.held) {
          glyph.vy += gravity * dt
          glyph.vx *= linearDamping
          glyph.vy *= linearDamping
          glyph.angularVelocity *= angularDamping
          glyph.ox += glyph.vx * dt
          glyph.oy += glyph.vy * dt
          glyph.angle += glyph.angularVelocity * dt
        }
      }

      for (let iteration = 0; iteration < 3; iteration++) {
        for (let i = 0; i < glyphs.length; i++) {
          for (let j = i + 1; j < glyphs.length; j++) {
            const a = glyphs[i], b = glyphs[j]
            const ax = a.x + a.ox, ay = a.y + a.oy, bx = b.x + b.ox, by = b.y + b.oy
            let dx = bx - ax, dy = by - ay
            let distance = Math.hypot(dx, dy)
            const support = (glyph: Glyph, directionX: number, directionY: number) => {
              const cosine = Math.cos(glyph.angle), sine = Math.sin(glyph.angle)
              const localX = cosine * directionX + sine * directionY
              const localY = -sine * directionX + cosine * directionY
              return 1 / Math.sqrt((localX / glyph.rx) ** 2 + (localY / glyph.ry) ** 2)
            }
            if (distance < .001) { dx = 1; dy = 0; distance = 1 }
            const nx = dx / distance, ny = dy / distance
            const minimum = support(a, nx, ny) + support(b, -nx, -ny)
            if (distance >= minimum) continue
            const invA = a.held ? 0 : 1 / a.mass
            const invB = b.held ? 0 : 1 / b.mass
            const invSum = invA + invB
            if (invSum <= 0) continue
            const correction = Math.max(0, minimum - distance - .35) * .72 / invSum
            a.ox -= nx * correction * invA; a.oy -= ny * correction * invA
            b.ox += nx * correction * invB; b.oy += ny * correction * invB
            const relativeX = b.vx - a.vx, relativeY = b.vy - a.vy
            const alongNormal = relativeX * nx + relativeY * ny
            if (alongNormal < 0) {
              const restitution = smoothImpact(-alongNormal, 180, 1400, .12, .58)
              const impulseLimit = 2200 * Math.min(devicePixelRatio, 1.25)
              const impulse = Math.min(impulseLimit, -(1 + restitution) * alongNormal / invSum)
              const impulseX = impulse * nx, impulseY = impulse * ny
              if (!a.held) { a.vx -= impulseX * invA; a.vy -= impulseY * invA }
              if (!b.held) { b.vx += impulseX * invB; b.vy += impulseY * invB }
              const tangentSpeed = relativeX * -ny + relativeY * nx
              if (!a.held) a.angularVelocity -= tangentSpeed / Math.max(1, a.r) * .055
              if (!b.held) b.angularVelocity += tangentSpeed / Math.max(1, b.r) * .055
            }
          }
        }
      }

      for (const glyph of glyphs) {
        let cx = glyph.x + glyph.ox
        let cy = glyph.y + glyph.oy
        const cosine = Math.abs(Math.cos(glyph.angle)), sine = Math.abs(Math.sin(glyph.angle))
        const extentX = cosine * glyph.rx + sine * glyph.ry
        const extentY = sine * glyph.rx + cosine * glyph.ry
        if (cx < extentX) { glyph.ox += extentX - cx; glyph.vx = Math.abs(glyph.vx) * smoothImpact(Math.abs(glyph.vx), 180, 1500, .12, .5); glyph.angularVelocity *= .82; cx = extentX }
        if (cx > canvas.width - extentX) { glyph.ox -= cx - (canvas.width - extentX); glyph.vx = -Math.abs(glyph.vx) * smoothImpact(Math.abs(glyph.vx), 180, 1500, .12, .5); glyph.angularVelocity *= .82 }
        if (cy < extentY) { glyph.oy += extentY - cy; glyph.vy = Math.abs(glyph.vy) * smoothImpact(Math.abs(glyph.vy), 200, 1600, .1, .44); cy = extentY }
        if (cy > canvas.height - extentY) {
          glyph.oy -= cy - (canvas.height - extentY)
          if (glyph.vy > 0) glyph.vy *= -smoothImpact(glyph.vy, 200, 1600, .1, .44)
          glyph.angularVelocity += glyph.vx / Math.max(1, glyph.r) * .025
          glyph.vx *= .82
          glyph.angularVelocity *= .74
        }
        glyph.vx = Math.max(-velocityLimit, Math.min(velocityLimit, glyph.vx))
        glyph.vy = Math.max(-velocityLimit, Math.min(velocityLimit, glyph.vy))
      }
    }

    function simulatePhysics(now: number) {
      const elapsed = Math.min(.05, Math.max(0, (now - physicsAt) / 1000))
      physicsAt = now
      physicsAccumulator += elapsed
      const fixedStep = 1 / 60
      let steps = 0
      while (physicsAccumulator >= fixedStep && steps < 4) {
        stepPhysics(fixedStep)
        physicsAccumulator -= fixedStep
        steps++
      }
    }

    function blurPass(source: WebGLTexture, target: WebGLTexture, directionX: number, directionY: number, radius: number, sourceWidth: number, sourceHeight: number) {
      gl!.bindFramebuffer(gl!.FRAMEBUFFER, framebuffer)
      gl!.framebufferTexture2D(gl!.FRAMEBUFFER, gl!.COLOR_ATTACHMENT0, gl!.TEXTURE_2D, target, 0)
      gl!.viewport(0, 0, halfWidth, halfHeight)
      bindProgram(blurProgram)
      gl!.activeTexture(gl!.TEXTURE3)
      gl!.bindTexture(gl!.TEXTURE_2D, source)
      gl!.uniform1i(gl!.getUniformLocation(blurProgram, 'u_source'), 3)
      gl!.uniform2f(gl!.getUniformLocation(blurProgram, 'u_texel'), 1 / sourceWidth, 1 / sourceHeight)
      gl!.uniform2f(gl!.getUniformLocation(blurProgram, 'u_direction'), directionX, directionY)
      gl!.uniform1f(gl!.getUniformLocation(blurProgram, 'u_radius'), radius)
      gl!.drawArrays(gl!.TRIANGLES, 0, 3)
    }

    function render(now: number) {
      resize(); simulatePhysics(now); drawMask()
      const video = videoRef.current
      const cameraReady = cameraReadyRef.current && !!video && video.readyState >= 2 && cameraChoice !== 'none'
      if (cameraReady) {
        gl!.activeTexture(gl!.TEXTURE0)
        gl!.bindTexture(gl!.TEXTURE_2D, bgTexture)
        gl!.pixelStorei(gl!.UNPACK_FLIP_Y_WEBGL, true)
        gl!.texImage2D(gl!.TEXTURE_2D, 0, gl!.RGBA, gl!.RGBA, gl!.UNSIGNED_BYTE, video!)
        gl!.pixelStorei(gl!.UNPACK_FLIP_Y_WEBGL, false)
      }
      let source = maskTexture
      const radii = [1, 1.75, 2.6]
      for (let index = 0; index < radii.length; index++) {
        blurPass(source, blurA, 1, 0, radii[index], index === 0 ? canvas.width : halfWidth, index === 0 ? canvas.height : halfHeight)
        blurPass(blurA, blurB, 0, 1, radii[index], halfWidth, halfHeight)
        source = blurB
      }
      const p = paramsRef.current
      gl!.bindFramebuffer(gl!.FRAMEBUFFER, null)
      gl!.viewport(0, 0, canvas.width, canvas.height)
      bindProgram(program)
      gl!.activeTexture(gl!.TEXTURE0); gl!.bindTexture(gl!.TEXTURE_2D, bgTexture)
      gl!.activeTexture(gl!.TEXTURE1); gl!.bindTexture(gl!.TEXTURE_2D, maskTexture)
      gl!.activeTexture(gl!.TEXTURE2); gl!.bindTexture(gl!.TEXTURE_2D, blurB)
      const loc = (name: string) => gl!.getUniformLocation(program, name)
      gl!.uniform1i(loc('u_background'), 0); gl!.uniform1i(loc('u_mask'), 1); gl!.uniform1i(loc('u_height'), 2)
      gl!.uniform2f(loc('u_resolution'), canvas.width, canvas.height)
      gl!.uniform2f(loc('u_videoSize'), video?.videoWidth || 16, video?.videoHeight || 9)
      gl!.uniform1f(loc('u_time'), now / 1000); gl!.uniform1f(loc('u_camera'), cameraReady ? 1 : 0)
      gl!.uniform1f(loc('u_thresh'), p.thresh); gl!.uniform1f(loc('u_wobble'), p.wobble); gl!.uniform1f(loc('u_bump'), p.bump)
      gl!.uniform1f(loc('u_refract'), p.refract); gl!.uniform1f(loc('u_disperse'), p.disperse); gl!.uniform1f(loc('u_fresnel'), p.fresnel)
      gl!.uniform1f(loc('u_frost'), p.frost); gl!.uniform1f(loc('u_irid'), p.irid); gl!.uniform1f(loc('u_shadow'), p.shadow)
      gl!.uniform1f(loc('u_fire'), styleRef.current === 'fire' ? 1 : 0)
      gl!.drawArrays(gl!.TRIANGLES, 0, 3)
      frames++
      if (now - fpsAt > 500) { setFps(Math.round(frames * 1000 / (now - fpsAt))); frames = 0; fpsAt = now }
      raf = requestAnimationFrame(render)
    }
    document.fonts.ready.then(() => { layoutKey = '' })
    raf = requestAnimationFrame(render)
    return () => {
      cancelAnimationFrame(raf)
      gl.deleteProgram(program); gl.deleteProgram(blurProgram); gl.deleteTexture(bgTexture); gl.deleteTexture(maskTexture)
      gl.deleteTexture(blurA); gl.deleteTexture(blurB); gl.deleteFramebuffer(framebuffer); gl.deleteBuffer(quad)
    }
  }, [cameraChoice])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (document.activeElement === inputRef.current && event.key.toLowerCase() !== 'escape') return
      if (event.key.toLowerCase() === 'h') setHidden((value) => !value)
      if (event.key.toLowerCase() === 'r') reset()
      if (event.key.toLowerCase() === 's') {
        const link = document.createElement('a')
        link.download = 'liquid-glass.png'
        link.href = canvasRef.current?.toDataURL('image/png') || ''
        link.click()
      }
    }
    addEventListener('keydown', onKey)
    return () => removeEventListener('keydown', onKey)
  }, [reset])

  return (
    <main className="fixed inset-0 overflow-hidden bg-[#0a0e14] text-white">
      <canvas ref={canvasRef} onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={() => releaseDrag('pointer')} onPointerCancel={() => releaseDrag('pointer')} className="absolute inset-0 h-full w-full touch-none" aria-label="Real-time liquid glass rendering; pinch or drag individual letters" />
      <video ref={videoRef} className="hidden" muted playsInline />

      <AnimatePresence>
        {!hidden && handPoints && (
          <>
            {[handPoints.thumb, handPoints.index].map((point, index) => (
              <motion.div key={index} initial={{ opacity: 0, scale: 1.3 }} animate={{ opacity: 1, scale: handPoints.pinch ? .68 : 1, left: `${point.x * 100}%`, top: `${point.y * 100}%` }} exit={{ opacity: 0 }} transition={{ duration: .08 }} className={`pointer-events-none fixed z-[18] h-7 w-7 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 ${handPoints.pinch ? 'border-cyan-200 bg-cyan-200/20 shadow-[0_0_14px_rgba(165,243,252,.8)]' : 'border-white/90 bg-white/5 shadow-[0_0_8px_rgba(0,0,0,.45)]'}`} />
            ))}
          </>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {!hidden && (
          <motion.aside
            initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}
            className="fixed right-4 top-4 z-20 w-[232px] overflow-hidden rounded-[14px] border border-white/20 bg-[rgba(16,22,32,.48)] font-bold text-[10px] tracking-[.06em] text-white/90 shadow-[0_8px_32px_rgba(0,0,0,.3)] backdrop-blur-2xl backdrop-saturate-150 max-[430px]:right-4"
          >
            <button type="button" onClick={() => setCollapsed((value) => !value)} className="flex w-full items-center justify-between px-[14px] py-[10px] text-left text-[11px] font-extrabold tracking-[.18em] hover:bg-white/[.06]">
              <span>CONTROLS</span><span>{collapsed ? '+' : '−'}</span>
            </button>
            <AnimatePresence initial={false}>
              {!collapsed && (
                <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                  <div className="px-[14px] pb-3 pt-1">
                    <label className="mb-[7px] grid grid-cols-[82px_1fr] items-center gap-2">
                      <span>camera</span>
                      <select value={cameraChoice} onChange={(event) => setCameraChoice(event.target.value)} className="min-w-0 rounded-[7px] border border-white/20 bg-white/10 px-2 py-[5px] text-[10px] text-white outline-none">
                        <option value="camera" className="text-black">{cameraName}</option>
                        <option value="none" className="text-black">none (sky)</option>
                      </select>
                    </label>
                    <label className="mb-[7px] grid grid-cols-[82px_1fr] items-center gap-2">
                      <span>style</span>
                      <select value={style} onChange={(event) => setStyle(event.target.value)} className="min-w-0 rounded-[7px] border border-white/20 bg-white/10 px-2 py-[5px] text-[10px] text-white outline-none">
                        <option value="liquid glass" className="text-black">liquid glass</option>
                        <option value="fire" className="text-black">fire</option>
                      </select>
                    </label>
                    {controls.map(([key, label, min, max, step, digits]) => (
                      <label key={key} className="my-[7px] grid grid-cols-[82px_1fr_34px] items-center gap-2">
                        <span>{label}</span>
                        <input aria-label={label} type="range" min={min} max={max} step={step} value={params[key]} onChange={(event) => setParams((value) => ({ ...value, [key]: Number(event.target.value) }))} className="glass-range w-full" />
                        <output className="text-right tabular-nums text-white/75">{params[key].toFixed(digits)}</output>
                      </label>
                    ))}
                    <button type="button" onClick={reset} className="mt-[10px] w-full rounded-lg border border-white/20 bg-white/10 py-[7px] text-[10px] font-extrabold tracking-[.18em] hover:bg-white/[.18]">RESET</button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.aside>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {!hidden && cameraNotice && (
          <motion.div initial={{ opacity: 0, y: -7 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -7 }} className="fixed left-1/2 top-4 z-10 -translate-x-1/2 rounded-[12px] border border-white/20 bg-[rgba(16,22,32,.48)] px-4 py-2 text-[9px] font-bold uppercase tracking-[.14em] text-white/80 shadow-[0_8px_24px_rgba(0,0,0,.28)] backdrop-blur-xl">
            {cameraNotice}
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {!hidden && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="pointer-events-none fixed inset-0 z-10 font-extrabold text-[10px] uppercase leading-[1.8] tracking-[.14em] text-white/85 [text-shadow:0_1px_8px_rgba(0,0,0,.5)]">
            <div className="absolute bottom-[18px] left-5">
              <div>H · HIDE UI</div><div>S · SCREENSHOT</div><div>R · RESET LETTERS</div><div>🤏 PINCH · DRAG</div><div className="mt-1 text-white/60">HAND TRACKING · {handStatus}</div>
            </div>
            <div className="absolute bottom-[18px] right-5">FPS · {fps}</div>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {!hidden && (
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 12 }} className="fixed bottom-[34px] left-1/2 z-30 w-[min(480px,calc(100vw-48px))] -translate-x-1/2">
            <input ref={inputRef} value={text} maxLength={26} onChange={(event) => setText(event.target.value)} aria-label="Liquid glass text" className="w-full rounded-full border border-white/30 bg-white/10 px-[22px] py-[14px] text-center text-[17px] font-extrabold uppercase tracking-[.06em] text-white shadow-[0_8px_32px_rgba(0,0,0,.35),inset_0_1px_0_rgba(255,255,255,.35)] outline-none backdrop-blur-2xl backdrop-saturate-150 transition focus:border-white/50 focus:bg-white/15" />
          </motion.div>
        )}
      </AnimatePresence>
    </main>
  )
}

export default App
