import { AnimatePresence, motion } from 'motion/react'
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision'
import { useCallback, useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'

type Glyph = {
  id: number
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
  spinUntil: number
  mass: number
  held: boolean
  alive: boolean
  respawnAt: number
}

type Drag = { index: number; dx: number; dy: number; source: 'hand' | 'pointer'; lastX: number; lastY: number; lastTime: number }
type HandPoint = { x: number; y: number }
type StyleMode = 'liquid glass' | 'bubble' | 'fire'
type PopDroplet = { angle: number; speed: number; size: number; hue: number }
type PopEffect = { ch: string; x: number; y: number; angle: number; fontSize: number; startedAt: number; droplets: PopDroplet[] }
type ContactState = { startedAt: number; cooldownUntil: number }

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
  thresh: 0.545,
  wobble: 0,
  bump: 3.1,
  refract: 0.406,
  disperse: 0.102,
  fresnel: 2,
  frost: 0.0515,
  irid: 1.5,
  shadow: 0.4,
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

const warpFragmentShader = `#version 300 es
precision highp float;
uniform sampler2D u_source;
uniform float u_time;
uniform float u_wobble;
uniform float u_aspect;
in vec2 v_uv;
out vec4 outColor;
float hash(vec2 p) {
  p = fract(p * vec2(234.34, 435.345));
  p += dot(p, p + 34.23);
  return fract(p.x * p.y);
}
float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3. - 2. * f);
  return mix(mix(hash(i), hash(i + vec2(1,0)), f.x), mix(hash(i + vec2(0,1)), hash(i + vec2(1)), f.x), f.y);
}
void main() {
  vec2 p = v_uv * vec2(u_aspect, 1.);
  float t = u_time * .6;
  vec2 warp = vec2(
    noise(p * 5. + vec2(t, t * .7)) - .5 + .3 * sin(p.y * 12. + t * 2.1),
    noise(p * 5. + vec2(-t * .8, t) + 17.3) - .5 + .3 * sin(p.x * 11. - t * 1.7)
  );
  float mask = texture(u_source, v_uv + warp * u_wobble).r;
  outColor = vec4(mask, mask, 0., 1.);
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
uniform float u_bump;
uniform float u_refract;
uniform float u_disperse;
uniform float u_fresnel;
uniform float u_frost;
uniform float u_irid;
uniform float u_shadow;
uniform float u_fire;
uniform float u_bubble;
in vec2 v_uv;
out vec4 outColor;

const float TAU = 6.28318530718;
float hash(vec2 p) {
  p = fract(p * vec2(234.34, 435.345));
  p += dot(p, p + 34.23);
  return fract(p.x * p.y);
}
float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3. - 2. * f);
  return mix(mix(hash(i), hash(i + vec2(1,0)), f.x), mix(hash(i + vec2(0,1)), hash(i + vec2(1)), f.x), f.y);
}
float fbm(vec2 p) {
  float value = 0., amplitude = .5;
  for (int i = 0; i < 5; i++) { value += amplitude * noise(p); p = p * 2.03 + 11.7; amplitude *= .5; }
  return value;
}
vec3 sky(vec2 uv) {
  vec3 col = mix(vec3(.62,.8,.98), vec3(.16,.42,.86), clamp(uv.y * 1.1, 0., 1.));
  vec2 p = uv * vec2(u_resolution.x / u_resolution.y, 1.) * 2.6 + vec2(u_time * .02, 0.);
  float cloudNoise = fbm(p + fbm(p * 1.7) * .6);
  float cloud = smoothstep(.48, .78, cloudNoise);
  col = mix(col, vec3(.98,.99,1.), cloud * .92);
  col = mix(col, vec3(.72,.8,.92), cloud * (1. - smoothstep(.55,.95,cloudNoise)) * .35);
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
vec3 bgBlur(vec2 uv, float radius) {
  if (radius < .0005) return bg(uv);
  vec2 px = radius / u_resolution * u_resolution.y;
  vec3 sum = bg(uv) * .294;
  sum += bg(uv + vec2(.85,.26) * px) * .1765;
  sum += bg(uv + vec2(-.42,.79) * px) * .1765;
  sum += bg(uv + vec2(-.71,-.54) * px) * .1765;
  sum += bg(uv + vec2(.38,-.83) * px) * .1765;
  return sum;
}
float profile(float height) {
  float threshold = min(.82, u_thresh + (u_bubble > .5 ? .018 : 0.));
  float t = clamp((height - threshold) / (1. - threshold), 0., 1.);
  float k = 1. - t;
  return sqrt(max(1. - k * k, 0.));
}
float heightAt(vec2 uv) { return profile(texture(u_height, uv).r); }
float softbox(vec3 reflected) {
  float band1 = smoothstep(.12,.32,reflected.y) * smoothstep(.62,.42,reflected.y);
  float band2 = smoothstep(-.55,-.4,reflected.y) * smoothstep(-.18,-.34,reflected.y) * .35;
  float side = smoothstep(.5,.9,abs(reflected.x)) * .15;
  return band1 + band2 + side;
}
void main() {
  vec2 uv = v_uv;
  float aspect = u_resolution.x / u_resolution.y;
  float rawHeight = texture(u_height, uv).r + (hash(uv * u_resolution) - .5) * .004;
  float antialias = fwidth(rawHeight) * 1.2 + .002;
  float threshold = min(.82, u_thresh + (u_bubble > .5 ? .018 : 0.));
  float shape = smoothstep(threshold - antialias, threshold + antialias, rawHeight);
  vec3 background = bg(uv);
  if (u_shadow > .001) {
    float shadowHeight = profile(texture(u_height, uv + vec2(-.012,.016)).r);
    float shadowStrength = u_bubble > .5 ? u_shadow * .08 : u_shadow;
    background *= 1. - shadowStrength * shadowHeight * (1. - shape);
  }
  if (shape < .001) { outColor = vec4(pow(background, vec3(.98)), 1.); return; }

  float epsilon = 2. / u_resolution.y;
  float hx = heightAt(uv + vec2(epsilon / aspect, 0.)) - heightAt(uv - vec2(epsilon / aspect, 0.));
  float hy = heightAt(uv + vec2(0., epsilon)) - heightAt(uv - vec2(0., epsilon));
  vec3 normal = normalize(vec3(-hx * u_bump, -hy * u_bump, 2. * epsilon * 40.));
  float height = profile(rawHeight);
  float sharp = texture(u_mask, uv).r;

  float thickness = u_refract * (.35 + .65 * (1. - height));
  vec3 incident = vec3(0.,0.,-1.);
  vec3 refractG = refract(incident, normal, 1. / (1.5 + u_disperse));
  vec2 axis = vec2(1. / aspect, 1.);
  if (u_bubble > .5) {
    float bubbleEdge = pow(clamp(1. - normal.z, 0., 1.), .72);
    float drift = noise(uv * 18. + vec2(u_time * .07, -u_time * .05));
    float bubblePhase = (1. - height) * 3.4 + bubbleEdge * 1.8 + drift * .38;
    vec3 membrane = .5 + .5 * cos(TAU * bubblePhase + vec3(0., 2.1, 4.2));
    vec3 transmitted = bg(uv + refractG.xy * thickness * axis * .34);
    vec3 view = vec3(0.,0.,1.);
    vec3 reflected = reflect(incident, normal);
    float bubbleFresnel = min((.025 + .975 * pow(clamp(1. - normal.z, 0., 1.), 5.)) * u_fresnel, 1.);
    vec3 environment = bg(clamp(uv + reflected.xy * .13 * axis, 0., 1.)) * .16 + vec3(softbox(reflected)) * .44;
    vec3 sunDir = normalize(vec3(-.58, .68, .46));
    float sun = pow(max(dot(normal, normalize(sunDir + view)), 0.), 150.) * 1.75;
    float softSun = pow(max(dot(normal, normalize(vec3(-.35, .62, .70) + view)), 0.), 28.) * .16;
    vec3 bubbleGlass = transmitted * .94
      + membrane * (bubbleEdge * .34 + .025) * (1.05 + u_irid * .16)
      + environment * bubbleFresnel
      + vec3(1., .91, .72) * sun
      + vec3(.82, .94, 1.) * softSun;
    outColor = vec4(pow(mix(background, bubbleGlass, shape), vec3(.98)), 1.);
    return;
  }

  vec3 refractR = refract(incident, normal, 1. / 1.5);
  vec3 refractB = refract(incident, normal, 1. / (1.5 + u_disperse * 2.));
  float frost = u_frost * (1. - sharp * .75);
  vec3 refracted;
  refracted.r = bgBlur(uv + refractR.xy * thickness * axis, frost).r;
  refracted.g = bgBlur(uv + refractG.xy * thickness * axis, frost).g;
  refracted.b = bgBlur(uv + refractB.xy * thickness * axis, frost).b;
  refracted *= mix(vec3(1.), vec3(.93,.98,1.), height * .6);

  float rim = pow(clamp(1. - normal.z, 0., 1.), 1.5);
  refracted *= 1. - rim * .16;
  float fresnel = min((.03 + .97 * pow(clamp(1. - normal.z, 0., 1.), 5.)) * u_fresnel, 1.);
  vec3 reflected = reflect(incident, normal);
  vec3 environment = bgBlur(clamp(uv + reflected.xy * .22 * axis, 0., 1.), .02) * .55 + vec3(softbox(reflected)) * .95;
  float phase = (1. - height) * 2.2 + rim * 1.3;
  vec3 film = .5 + .5 * cos(TAU * phase + vec3(0.,2.1,4.2));
  vec3 iridescence = film * u_irid * rim * (.4 + .6 * noise(uv * 30. + u_time * .15));
  vec3 view = vec3(0.,0.,1.);
  vec3 light1 = normalize(vec3(-.45,.65,.62));
  vec3 light2 = normalize(vec3(.55,.42,.72));
  float spec1 = pow(max(dot(normal, normalize(light1 + view)), 0.), 90.) * .9;
  float spec2 = pow(max(dot(normal, normalize(light2 + view)), 0.), 420.) * 1.6;
  vec3 glass = refracted * (1. - fresnel * .7) + environment * fresnel + iridescence * (.35 + fresnel) + vec3(spec1 + spec2);
  if (u_fire > .5) {
    glass = mix(vec3(.9,.08,0.), vec3(1.,.75,0.), height) + vec3(spec1 + spec2);
  }
  outColor = vec4(pow(mix(background, glass, shape), vec3(.98)), 1.);
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
  float weights[7];
  weights[0]=.1964; weights[1]=.1747; weights[2]=.1216; weights[3]=.0661;
  weights[4]=.0281; weights[5]=.0093; weights[6]=.0024;
  float value = texture(u_source, v_uv).r * weights[0];
  for (int i=1; i<7; i++) {
    float offset = float(i);
    value += texture(u_source, v_uv + stepUv * offset).r * weights[i];
    value += texture(u_source, v_uv - stepUv * offset).r * weights[i];
  }
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
  const fxCanvasRef = useRef<HTMLCanvasElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const paramsRef = useRef(defaults)
  const textRef = useRef('LIQUID GLASS')
  const cameraReadyRef = useRef(false)
  const cameraChoiceRef = useRef('camera')
  const styleRef = useRef<StyleMode>('liquid glass')
  const glyphsRef = useRef<Glyph[]>([])
  const dragRef = useRef<Drag | null>(null)
  const popEffectsRef = useRef<PopEffect[]>([])
  const contactsRef = useRef(new Map<string, ContactState>())
  const resetEpochRef = useRef(0)
  const [params, setParams] = useState(defaults)
  const [text, setText] = useState('LIQUID GLASS')
  const [collapsed, setCollapsed] = useState(false)
  const [hidden, setHidden] = useState(false)
  const [fps, setFps] = useState(60)
  const [cameraNotice, setCameraNotice] = useState('')
  const [cameraChoice, setCameraChoice] = useState('camera')
  const [cameraName, setCameraName] = useState('camera')
  const [style, setStyle] = useState<StyleMode>('liquid glass')
  const [handStatus, setHandStatus] = useState<'LOADING' | 'READY' | 'NO HAND'>('LOADING')
  const [handPoints, setHandPoints] = useState<{ thumb: HandPoint; index: HandPoint; pinch: boolean } | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => { paramsRef.current = params }, [params])
  useEffect(() => { textRef.current = text }, [text])
  useEffect(() => {
    const previous = styleRef.current
    styleRef.current = style
    if (previous === style) return
    contactsRef.current.clear()
    popEffectsRef.current = []
    const lift = 80 * Math.min(devicePixelRatio, 1)
    for (const glyph of glyphsRef.current) {
      glyph.alive = true
      glyph.respawnAt = 0
      glyph.held = false
      glyph.angularVelocity = 0
      glyph.spinUntil = 0
      glyph.vy = style === 'bubble'
        ? -Math.max(lift, Math.abs(glyph.vy) * .65)
        : Math.max(lift, Math.abs(glyph.vy) * .65)
    }
    dragRef.current = null
  }, [style])
  useEffect(() => { cameraChoiceRef.current = cameraChoice }, [cameraChoice])

  const reset = useCallback(() => {
    setParams(defaults)
    setText('LIQUID GLASS')
    resetEpochRef.current++
    contactsRef.current.clear()
    popEffectsRef.current = []
    for (const glyph of glyphsRef.current) {
      glyph.ox = 0; glyph.oy = 0; glyph.vx = 0; glyph.vy = 0; glyph.angle = 0; glyph.angularVelocity = 0; glyph.spinUntil = 0; glyph.held = false; glyph.alive = true; glyph.respawnAt = 0
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

  const glyphIndexAt = useCallback((x: number, y: number) => {
    let best = -1
    let bestDistance = Infinity
    glyphsRef.current.forEach((glyph, index) => {
      if (!glyph.alive || !glyph.ch.trim()) return
      const dx = x - glyph.x - glyph.ox
      const dy = y - glyph.y - glyph.oy
      const cosine = Math.cos(glyph.angle), sine = Math.sin(glyph.angle)
      const localX = cosine * dx + sine * dy
      const localY = -sine * dx + cosine * dy
      const hit = (localX / Math.max(1, glyph.width * .58)) ** 2 + (localY / Math.max(1, glyph.fontSize * .43)) ** 2
      const distance = Math.hypot(dx, dy)
      if (hit <= 1.25 && distance < bestDistance) { best = index; bestDistance = distance }
    })
    return best
  }, [])

  const pickGlyph = useCallback((x: number, y: number, source: Drag['source']) => {
    const best = glyphIndexAt(x, y)
    if (best < 0) return false
    releaseDrag()
    const glyph = glyphsRef.current[best]
    glyph.held = true
    glyph.vx = 0
    glyph.vy = 0
    glyph.angularVelocity = 0
    glyph.spinUntil = 0
    dragRef.current = { index: best, dx: glyph.x + glyph.ox - x, dy: glyph.y + glyph.oy - y, source, lastX: x, lastY: y, lastTime: performance.now() }
    return true
  }, [glyphIndexAt, releaseDrag])

  const popGlyphAt = useCallback((x: number, y: number) => {
    if (styleRef.current !== 'bubble') return false
    const index = glyphIndexAt(x, y)
    if (index < 0) return false
    const glyph = glyphsRef.current[index]
    const now = performance.now()
    glyph.alive = false
    glyph.held = false
    glyph.respawnAt = now + 3350
    glyph.vx = 0
    glyph.vy = 0
    if (dragRef.current?.index === index) dragRef.current = null
    const droplets = Array.from({ length: 12 }, (_, dropletIndex): PopDroplet => ({
      angle: (Math.PI * 2 * dropletIndex) / 12 + (Math.random() - .5) * .28,
      speed: glyph.fontSize * (.32 + Math.random() * .38),
      size: Math.max(1.5, glyph.fontSize * (.008 + Math.random() * .012)),
      hue: [184, 312, 48][dropletIndex % 3],
    }))
    popEffectsRef.current.push({ ch: glyph.ch, x: glyph.x + glyph.ox, y: glyph.y + glyph.oy, angle: glyph.angle, fontSize: glyph.fontSize, startedAt: now, droplets })
    contactsRef.current.clear()
    return true
  }, [glyphIndexAt])

  const moveDrag = useCallback((x: number, y: number, source: Drag['source']) => {
    const drag = dragRef.current
    if (!drag || drag.source !== source) return
    const glyph = glyphsRef.current[drag.index]
    if (!glyph) return
    const now = performance.now()
    const dt = Math.max(.008, Math.min(.08, (now - drag.lastTime) / 1000))
    const maxThrowSpeed = 2400 * Math.min(devicePixelRatio, 1)
    const sampleVx = Math.max(-maxThrowSpeed, Math.min(maxThrowSpeed, (x - drag.lastX) / dt))
    const sampleVy = Math.max(-maxThrowSpeed, Math.min(maxThrowSpeed, (y - drag.lastY) / dt))
    glyph.vx += (sampleVx - glyph.vx) * .42
    glyph.vy += (sampleVy - glyph.vy) * .42
    if (Math.hypot(sampleVx, sampleVy) > 80 * Math.min(devicePixelRatio, 1)) {
      glyph.spinUntil = now + 2000
      glyph.angularVelocity += ((sampleVx / Math.max(1, glyph.r)) * .28 - glyph.angularVelocity) * .18
      glyph.angularVelocity = Math.max(-8, Math.min(8, glyph.angularVelocity))
    }
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

  const onDoubleClick = useCallback((event: ReactPointerEvent<HTMLCanvasElement>) => {
    const point = pointerPosition(event)
    releaseDrag('pointer')
    popGlyphAt(point.x, point.y)
  }, [pointerPosition, popGlyphAt, releaseDrag])

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
    let previousIndexSample: { x: number; y: number; time: number; glyph: number } | null = null
    let pokeCooldownUntil = 0
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
      if (landmarker && video && canvas && cameraReadyRef.current && cameraChoiceRef.current === 'camera' && video.readyState >= 2 && now - lastDetection >= 67) {
        lastDetection = now
        const videoWidth = Math.max(1, video.videoWidth)
        const videoHeight = Math.max(1, video.videoHeight)
        const inferenceScale = Math.min(1, 512 / Math.max(videoWidth, videoHeight))
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
          const indexPx = indexRaw.x * canvas.width
          const indexPy = indexRaw.y * canvas.height
          const indexGlyph = glyphIndexAt(indexPx, indexPy)
          if (styleRef.current === 'bubble' && !nextPinching && previousIndexSample && now >= pokeCooldownUntil) {
            const sampleDt = Math.max(.02, (now - previousIndexSample.time) / 1000)
            const motionX = indexPx - previousIndexSample.x
            const motionY = indexPy - previousIndexSample.y
            const motionLength = Math.hypot(motionX, motionY)
            const speed = motionLength / sampleDt
            const pokeThreshold = Math.min(canvas.width, canvas.height) * 1.1
            const target = glyphsRef.current[indexGlyph]
            const targetX = target ? target.x + target.ox - previousIndexSample.x : 0
            const targetY = target ? target.y + target.oy - previousIndexSample.y : 0
            const targetDistance = Math.hypot(targetX, targetY)
            const approach = motionLength > 0 && targetDistance > 0 ? (motionX * targetX + motionY * targetY) / (motionLength * targetDistance) : 0
            if (indexGlyph >= 0 && previousIndexSample.glyph !== indexGlyph && speed >= pokeThreshold && approach > .72 && popGlyphAt(indexPx, indexPy)) {
              pokeCooldownUntil = now + 650
            }
          }
          previousIndexSample = { x: indexPx, y: indexPy, time: now, glyph: indexGlyph }
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
          previousIndexSample = null
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
  }, [glyphIndexAt, moveDrag, pickGlyph, popGlyphAt, releaseDrag])

  useEffect(() => {
    const canvas = canvasRef.current!
    const fxCanvas = fxCanvasRef.current!
    const fxCtx = fxCanvas.getContext('2d')!
    const gl = canvas.getContext('webgl2', { alpha: false, antialias: true })
    if (!gl) { setCameraNotice('WebGL2 unavailable'); return }
    const program = createProgram(gl, fragmentShader)
    const warpProgram = createProgram(gl, warpFragmentShader)
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
    const warpTexture = gl.createTexture()!
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
    let fxWasActive = false
    let targetsReady = false

    function resize() {
      const dpr = Math.min(devicePixelRatio, 1)
      const w = Math.floor(innerWidth * dpr)
      const h = Math.floor(innerHeight * dpr)
      if (!targetsReady || canvas.width !== w || canvas.height !== h) {
        canvas.width = maskCanvas.width = w
        canvas.height = maskCanvas.height = h
        fxCanvas.width = w
        fxCanvas.height = h
        halfWidth = Math.max(1, Math.floor(w / 2))
        halfHeight = Math.max(1, Math.floor(h / 2))
        for (const texture of [warpTexture, blurA, blurB]) {
          gl!.bindTexture(gl!.TEXTURE_2D, texture)
          configureTexture(texture)
          gl!.texImage2D(gl!.TEXTURE_2D, 0, gl!.RGBA8, halfWidth, halfHeight, 0, gl!.RGBA, gl!.UNSIGNED_BYTE, null)
        }
        targetsReady = true
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
      let glyphId = 0
      lines.slice(0,4).forEach((value, index) => {
        let x = (w - trackedWidth(value, tracking)) / 2
        for (const char of value) {
          const width = maskCtx.measureText(char).width
          const radiusX = Math.min(size * .38, Math.max(size * .13, width * .43))
          const radiusY = size * .36
          glyphs.push({
            id: glyphId++, ch: char, x: x + width / 2, y: startY + index * lineHeight, width, r: Math.sqrt(radiusX * radiusY), rx: radiusX, ry: radiusY, fontSize: size,
            ox: 0, oy: 0, vx: 0, vy: 0, angle: 0, angularVelocity: 0, spinUntil: 0,
            mass: Math.max(.65, width / Math.max(1, size)), held: false, alive: true, respawnAt: 0,
          })
          x += width + tracking
        }
      })
      glyphsRef.current = glyphs
      contactsRef.current.clear()
      popEffectsRef.current = []
    }

    function drawMask() {
      const p = paramsRef.current
      const nextKey = `${maskCanvas.width}:${maskCanvas.height}:${textRef.current}:${p.textScale}:${p.spacing}:${resetEpochRef.current}`
      if (nextKey !== layoutKey) { layoutKey = nextKey; rebuildLayout() }
      const w = maskCanvas.width, h = maskCanvas.height
      maskCtx.clearRect(0,0,w,h)
      maskCtx.fillStyle = 'white'
      maskCtx.lineJoin = 'round'
      maskCtx.textAlign = 'center'
      maskCtx.textBaseline = 'middle'
      for (const glyph of glyphsRef.current) {
        if (!glyph.alive || !glyph.ch.trim()) continue
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

    function stepPhysics(dt: number, now: number) {
      const bubble = styleRef.current === 'bubble'
      if (bubble) {
        for (const glyph of glyphsRef.current) {
          if (glyph.alive || !glyph.ch.trim() || now < glyph.respawnAt) continue
          const extentY = Math.max(glyph.rx, glyph.ry)
          glyph.alive = true
          glyph.respawnAt = 0
          glyph.ox = (Math.random() - .5) * glyph.fontSize * .22
          glyph.oy = canvas.height - extentY - canvas.height * .035 - glyph.y
          glyph.vx = (Math.random() - .5) * canvas.width * .11
          glyph.vy = -canvas.height * (.20 + Math.random() * .08)
          glyph.angle = 0
          glyph.angularVelocity = 0
          glyph.spinUntil = 0
        }
      }
      const glyphs = glyphsRef.current.filter((glyph) => glyph.alive && glyph.ch.trim())
      const gravity = canvas.height * (bubble ? -.34 : 1.35)
      const linearDamping = Math.pow(bubble ? .991 : .994, dt * 60)
      const angularDamping = Math.pow(bubble ? .975 : .98, dt * 60)
      const velocityLimit = 3200 * Math.min(devicePixelRatio, 1)
      const smoothImpact = (speed: number, low: number, high: number, minimum: number, maximum: number) => {
        const linear = Math.max(0, Math.min(1, (speed - low) / (high - low)))
        const eased = linear * linear * (3 - 2 * linear)
        return minimum + (maximum - minimum) * eased
      }
      for (const glyph of glyphs) {
        if (!glyph.held) {
          glyph.vy += gravity * dt
          if (bubble) {
            glyph.vx += Math.sin(now * .00072 + glyph.id * 1.91) * canvas.width * .018 * dt
          }
          glyph.vx *= linearDamping
          glyph.vy *= linearDamping
          glyph.ox += glyph.vx * dt
          glyph.oy += glyph.vy * dt
          if (now < glyph.spinUntil) {
            glyph.angularVelocity *= angularDamping
            glyph.angle += glyph.angularVelocity * dt
          } else {
            glyph.angularVelocity = 0
          }
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
            const relativeX = b.vx - a.vx, relativeY = b.vy - a.vy
            const impactSpeed = Math.hypot(relativeX, relativeY)
            const pairKey = a.id < b.id ? `${a.id}:${b.id}` : `${b.id}:${a.id}`
            if (bubble && iteration === 0) {
              const range = minimum * 1.12
              let contact = contactsRef.current.get(pairKey)
              if (distance < range) {
                if (!contact && impactSpeed < 900) {
                  contact = { startedAt: now, cooldownUntil: 0 }
                  contactsRef.current.set(pairKey, contact)
                }
                if (contact && now >= contact.cooldownUntil) {
                  const age = now - contact.startedAt
                  if (age < 450) {
                    const pull = Math.max(0, distance - minimum * .94) * 3.5 * dt
                    if (!a.held) { a.vx += nx * pull; a.vy += ny * pull }
                    if (!b.held) { b.vx -= nx * pull; b.vy -= ny * pull }
                    if (!a.held) { a.vx += relativeX * .015; a.vy += relativeY * .015 }
                    if (!b.held) { b.vx -= relativeX * .015; b.vy -= relativeY * .015 }
                  } else {
                    const separate = 55
                    if (!a.held) { a.vx -= nx * separate; a.vy -= ny * separate }
                    if (!b.held) { b.vx += nx * separate; b.vy += ny * separate }
                    contact.cooldownUntil = now + 650
                  }
                }
              } else if (contact && now >= contact.cooldownUntil) {
                contactsRef.current.delete(pairKey)
              }
            }
            if (distance >= minimum) continue
            const invA = a.held ? 0 : 1 / a.mass
            const invB = b.held ? 0 : 1 / b.mass
            const invSum = invA + invB
            if (invSum <= 0) continue
            const contact = contactsRef.current.get(pairKey)
            const sticking = bubble && !!contact && now >= contact.cooldownUntil && now - contact.startedAt < 450
            const correctionStrength = sticking ? .25 : .72
            const correction = Math.max(0, minimum - distance - .35) * correctionStrength / invSum
            a.ox -= nx * correction * invA; a.oy -= ny * correction * invA
            b.ox += nx * correction * invB; b.oy += ny * correction * invB
            const alongNormal = relativeX * nx + relativeY * ny
            if (alongNormal < 0) {
              const restitution = sticking ? .02 : bubble
                ? smoothImpact(-alongNormal, 160, 1100, .08, .38)
                : smoothImpact(-alongNormal, 180, 1400, .12, .58)
              const impulseLimit = 2200 * Math.min(devicePixelRatio, 1)
              const impulse = Math.min(impulseLimit, -(1 + restitution) * alongNormal / invSum)
              const impulseX = impulse * nx, impulseY = impulse * ny
              if (!a.held) { a.vx -= impulseX * invA; a.vy -= impulseY * invA }
              if (!b.held) { b.vx += impulseX * invB; b.vy += impulseY * invB }
              const externallyDriven = a.held || b.held || now < a.spinUntil || now < b.spinUntil
              if (externallyDriven && impactSpeed > 120 * Math.min(devicePixelRatio, 1)) {
                const tangentSpeed = relativeX * -ny + relativeY * nx
                if (!a.held) {
                  a.spinUntil = now + 1600
                  a.angularVelocity -= tangentSpeed / Math.max(1, a.r) * .055
                }
                if (!b.held) {
                  b.spinUntil = now + 1600
                  b.angularVelocity += tangentSpeed / Math.max(1, b.r) * .055
                }
              }
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
        const wallBounce = bubble ? .28 : smoothImpact(Math.abs(glyph.vx), 180, 1500, .12, .5)
        if (cx < extentX) { glyph.ox += extentX - cx; glyph.vx = Math.abs(glyph.vx) * wallBounce; cx = extentX }
        if (cx > canvas.width - extentX) { glyph.ox -= cx - (canvas.width - extentX); glyph.vx = -Math.abs(glyph.vx) * wallBounce }
        if (bubble) {
          const ceiling = canvas.height * .055 + extentY
          if (cy < ceiling) {
            const penetration = ceiling - cy
            glyph.oy += penetration * .28
            glyph.vy += penetration * 8 * dt
            if (glyph.vy < 0) glyph.vy *= .72
            cy = glyph.y + glyph.oy
          }
        } else if (cy < extentY) {
          glyph.oy += extentY - cy
          glyph.vy = Math.abs(glyph.vy) * smoothImpact(Math.abs(glyph.vy), 200, 1600, .1, .44)
          cy = extentY
        }
        if (cy > canvas.height - extentY) {
          glyph.oy -= cy - (canvas.height - extentY)
          if (glyph.vy > 0) glyph.vy *= -(bubble ? .18 : smoothImpact(glyph.vy, 200, 1600, .1, .44))
          glyph.vx *= .82
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
        stepPhysics(fixedStep, now)
        physicsAccumulator -= fixedStep
        steps++
      }
    }

    function drawPopEffects(now: number) {
      const active = popEffectsRef.current.filter((effect) => now - effect.startedAt < 350)
      popEffectsRef.current = active
      if (!active.length) {
        if (fxWasActive) {
          fxCtx.clearRect(0, 0, fxCanvas.width, fxCanvas.height)
          fxCanvas.style.display = 'none'
        }
        fxWasActive = false
        return
      }
      fxWasActive = true
      fxCanvas.style.display = 'block'
      fxCtx.clearRect(0, 0, fxCanvas.width, fxCanvas.height)
      for (const effect of active) {
        const age = (now - effect.startedAt) / 350
        const eased = 1 - Math.pow(1 - age, 3)
        const opacity = Math.pow(1 - age, 1.35)
        fxCtx.save()
        fxCtx.translate(effect.x, effect.y)
        fxCtx.rotate(effect.angle)
        fxCtx.scale(1 - eased * .36, 1 - eased * .36)
        fxCtx.font = `800 ${effect.fontSize}px "Baloo 2", "Arial Rounded MT Bold", sans-serif`
        fxCtx.textAlign = 'center'
        fxCtx.textBaseline = 'middle'
        fxCtx.lineJoin = 'round'
        fxCtx.globalCompositeOperation = 'screen'
        fxCtx.lineWidth = Math.max(1.2, effect.fontSize * .016)
        for (const [offset, color] of [[-1.4, `hsla(184,100%,82%,${opacity * .82})`], [0, `hsla(312,100%,84%,${opacity * .72})`], [1.4, `hsla(48,100%,83%,${opacity * .72})`]] as const) {
          fxCtx.save()
          fxCtx.translate(offset, 0)
          fxCtx.strokeStyle = color
          fxCtx.strokeText(effect.ch, 0, 0)
          fxCtx.restore()
        }
        fxCtx.restore()

        const seconds = (now - effect.startedAt) / 1000
        for (const droplet of effect.droplets) {
          const distance = droplet.speed * seconds
          const px = effect.x + Math.cos(droplet.angle) * distance
          const py = effect.y + Math.sin(droplet.angle) * distance + effect.fontSize * .7 * seconds * seconds
          fxCtx.beginPath()
          fxCtx.arc(px, py, droplet.size * (1 - age * .45), 0, Math.PI * 2)
          fxCtx.fillStyle = `hsla(${droplet.hue},100%,86%,${opacity * .66})`
          fxCtx.fill()
        }
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

    function warpPass(now: number) {
      gl!.bindFramebuffer(gl!.FRAMEBUFFER, framebuffer)
      gl!.framebufferTexture2D(gl!.FRAMEBUFFER, gl!.COLOR_ATTACHMENT0, gl!.TEXTURE_2D, warpTexture, 0)
      gl!.viewport(0, 0, halfWidth, halfHeight)
      bindProgram(warpProgram)
      gl!.activeTexture(gl!.TEXTURE3)
      gl!.bindTexture(gl!.TEXTURE_2D, maskTexture)
      gl!.uniform1i(gl!.getUniformLocation(warpProgram, 'u_source'), 3)
      gl!.uniform1f(gl!.getUniformLocation(warpProgram, 'u_time'), now / 1000)
      const wobble = styleRef.current === 'bubble' ? Math.max(paramsRef.current.wobble, .0035) : paramsRef.current.wobble
      gl!.uniform1f(gl!.getUniformLocation(warpProgram, 'u_wobble'), wobble)
      gl!.uniform1f(gl!.getUniformLocation(warpProgram, 'u_aspect'), canvas.width / canvas.height)
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
      warpPass(now)
      let source = warpTexture
      const fontSize = glyphsRef.current.find((glyph) => glyph.ch.trim())?.fontSize || canvas.height * .3
      const sizeFactor = Math.min(1, fontSize / (canvas.height * .3))
      const baseRadius = Math.max(.35, (halfHeight / 540) * Math.max(.35, sizeFactor))
      const radii = [baseRadius, baseRadius * 2, baseRadius * 3]
      for (let index = 0; index < radii.length; index++) {
        blurPass(source, blurA, 1, 0, radii[index], halfWidth, halfHeight)
        blurPass(blurA, blurB, 0, 1, radii[index], halfWidth, halfHeight)
        source = blurB
      }
      const p = paramsRef.current
      gl!.bindFramebuffer(gl!.FRAMEBUFFER, null)
      gl!.viewport(0, 0, canvas.width, canvas.height)
      bindProgram(program)
      gl!.activeTexture(gl!.TEXTURE0); gl!.bindTexture(gl!.TEXTURE_2D, bgTexture)
      gl!.activeTexture(gl!.TEXTURE1); gl!.bindTexture(gl!.TEXTURE_2D, warpTexture)
      gl!.activeTexture(gl!.TEXTURE2); gl!.bindTexture(gl!.TEXTURE_2D, blurB)
      const loc = (name: string) => gl!.getUniformLocation(program, name)
      gl!.uniform1i(loc('u_background'), 0); gl!.uniform1i(loc('u_mask'), 1); gl!.uniform1i(loc('u_height'), 2)
      gl!.uniform2f(loc('u_resolution'), canvas.width, canvas.height)
      gl!.uniform2f(loc('u_videoSize'), video?.videoWidth || 16, video?.videoHeight || 9)
      gl!.uniform1f(loc('u_time'), now / 1000); gl!.uniform1f(loc('u_camera'), cameraReady ? 1 : 0)
      gl!.uniform1f(loc('u_thresh'), p.thresh); gl!.uniform1f(loc('u_bump'), p.bump)
      gl!.uniform1f(loc('u_refract'), p.refract); gl!.uniform1f(loc('u_disperse'), p.disperse); gl!.uniform1f(loc('u_fresnel'), p.fresnel)
      gl!.uniform1f(loc('u_frost'), p.frost); gl!.uniform1f(loc('u_irid'), p.irid); gl!.uniform1f(loc('u_shadow'), p.shadow)
      gl!.uniform1f(loc('u_fire'), styleRef.current === 'fire' ? 1 : 0)
      gl!.uniform1f(loc('u_bubble'), styleRef.current === 'bubble' ? 1 : 0)
      gl!.drawArrays(gl!.TRIANGLES, 0, 3)
      drawPopEffects(now)
      frames++
      if (now - fpsAt > 500) { setFps(Math.round(frames * 1000 / (now - fpsAt))); frames = 0; fpsAt = now }
      raf = requestAnimationFrame(render)
    }
    document.fonts.ready.then(() => { layoutKey = '' })
    raf = requestAnimationFrame(render)
    return () => {
      cancelAnimationFrame(raf)
      gl.deleteProgram(program); gl.deleteProgram(warpProgram); gl.deleteProgram(blurProgram); gl.deleteTexture(bgTexture); gl.deleteTexture(maskTexture)
      gl.deleteTexture(warpTexture); gl.deleteTexture(blurA); gl.deleteTexture(blurB); gl.deleteFramebuffer(framebuffer); gl.deleteBuffer(quad)
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
      <canvas ref={canvasRef} onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={() => releaseDrag('pointer')} onPointerCancel={() => releaseDrag('pointer')} onDoubleClick={onDoubleClick} className="absolute inset-0 h-full w-full touch-none" aria-label="Real-time liquid glass and bubble rendering; pinch or drag letters, double click bubbles to pop" />
      <canvas ref={fxCanvasRef} className="pointer-events-none absolute inset-0 z-[5] hidden h-full w-full" aria-hidden="true" />
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
                      <select value={style} onChange={(event) => setStyle(event.target.value as StyleMode)} className="min-w-0 rounded-[7px] border border-white/20 bg-white/10 px-2 py-[5px] text-[10px] text-white outline-none">
                        <option value="liquid glass" className="text-black">liquid glass</option>
                        <option value="bubble" className="text-black">bubble</option>
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
              <div>H · HIDE UI</div><div>S · SCREENSHOT</div><div>R · RESET LETTERS</div><div>🤏 PINCH · DRAG</div>{style === 'bubble' && <div>☝ DOUBLE CLICK / QUICK POKE · POP</div>}<div className="mt-1 text-white/60">HAND TRACKING · {handStatus}</div>
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
