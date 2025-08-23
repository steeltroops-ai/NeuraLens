'use client';

import React, { useEffect, useRef } from 'react';

// Utility hook for high-DPI canvas
function useHiDpiCanvas() {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;

    const resize = () => {
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      const cw = Math.max(240, canvas.clientWidth);
      const ch = Math.max(160, canvas.clientHeight);
      canvas.width = Math.floor(cw * dpr);
      canvas.height = Math.floor(ch * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    return () => ro.disconnect();
  }, []);
  return ref;
}

function clamp(v: number, a: number, b: number) {
  return Math.min(b, Math.max(a, v));
}

// Simple Perlin noise implementation
function makePerlin() {
  const perm = new Uint8Array(512);
  const p = new Uint8Array(256);
  for (let i = 0; i < 256; i++) p[i] = i;
  // shuffle
  for (let i = 255; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const t = p[i]!;
    p[i] = p[j]!;
    p[j] = t;
  }
  for (let i = 0; i < 512; i++) perm[i] = p[i & 255]!;

  function fade(t: number) {
    return t * t * t * (t * (t * 6 - 15) + 10);
  }
  function lerp(a: number, b: number, t: number) {
    return a + t * (b - a);
  }
  function grad(hash: number, x: number, y: number) {
    const h = hash & 3;
    const u = h < 2 ? x : y;
    const v = h < 2 ? y : x;
    return (h & 1 ? -u : u) + (h & 2 ? -2 * v : 2 * v);
  }
  return function noise(x: number, y: number) {
    const X = Math.floor(x) & 255;
    const Y = Math.floor(y) & 255;
    const xf = x - Math.floor(x);
    const yf = y - Math.floor(y);
    const topRight = perm[perm[X + 1]! + Y + 1]!;
    const topLeft = perm[perm[X]! + Y + 1]!;
    const bottomRight = perm[perm[X + 1]! + Y]!;
    const bottomLeft = perm[perm[X]! + Y]!;
    const u = fade(xf);
    const v = fade(yf);
    const x1 = lerp(grad(bottomLeft, xf, yf), grad(bottomRight, xf - 1, yf), u);
    const x2 = lerp(
      grad(topLeft, xf, yf - 1),
      grad(topRight, xf - 1, yf - 1),
      u
    );
    return (lerp(x1, x2, v) + 1) / 2; // normalize 0..1
  };
}

const perlin = makePerlin();

// Speech Analysis - Animated Waveform Patterns
export function SpeechWaveform() {
  const ref = useHiDpiCanvas();
  useEffect(() => {
    const canvas = ref.current!;
    const ctx = canvas.getContext('2d')!;
    let raf = 0;
    let t = 0;

    // formant parameters (F1,F2,F3 typical ranges)
    const formants = [
      { f: 600, q: 6 },
      { f: 1200, q: 8 },
      { f: 2500, q: 10 },
    ];

    function render() {
      const W = Math.max(480, canvas.clientWidth);
      const H = Math.max(220, canvas.clientHeight);
      ctx.clearRect(0, 0, W, H);

      // vocal tract simplified: series of overlapping ellipses
      const cx = W / 2;
      const topY = H * 0.28;
      for (let i = 0; i < 5; i++) {
        const w = W * 0.7 * (1 - i * 0.09);
        const h = H * 0.16 * (1 - i * 0.06);
        ctx.beginPath();
        ctx.ellipse(cx - i * 8, topY + i * 6, w * 0.5, h, 0, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0, 122, 255, 0.03)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(0, 122, 255, 0.1)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // generate a time-domain waveform by summing three band-limited sinusoids
      const samples = 420;
      ctx.beginPath();
      for (let i = 0; i < samples; i++) {
        const u = i / (samples - 1);
        const x = W * 0.08 + u * (W * 0.84);
        // drive frequency modulated by slow vowels
        const baseFreq = 110 + 40 * Math.sin(t * 0.6 + u * 3.2);
        let yval = 0;
        // excite and filter through formant resonators
        for (let k = 0; k < formants.length; k++) {
          const F = formants[k]!.f * (1 + 0.1 * Math.sin(t * 0.4 + k));
          const amp = 1.0 / (1 + Math.abs(k - 1));
          yval +=
            amp *
            Math.sin((baseFreq + F) * (u * 0.02 + t * 0.001)) *
            (1 / (1 + Math.pow((F - 800 * (k + 1)) / 400, 2)));
        }
        const y = H * 0.66 + yval * 18;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = 'rgba(0, 122, 255, 0.4)';
      ctx.lineWidth = 2;
      ctx.stroke();

      // draw spectrogram-like formant bands
      for (let k = 0; k < formants.length; k++) {
        const F = formants[k]!.f;
        const bx = W * 0.15 + (k / (formants.length - 1)) * (W * 0.7);
        const energy = 0.08 + 0.06 * Math.abs(Math.sin(t * 0.5 + k));
        ctx.fillStyle = `rgba(0, 122, 255, ${0.2 + energy})`;
        ctx.fillRect(
          bx - 10,
          H * 0.82 - energy * H * 0.25,
          20,
          energy * H * 0.25
        );
      }

      t += 0.02;
      raf = requestAnimationFrame(render);
    }

    raf = requestAnimationFrame(render);
    return () => cancelAnimationFrame(raf);
  }, [ref]);

  return <canvas ref={ref} className="absolute inset-0 h-full w-full" />;
}

// Retinal Analysis - Animated Eye with Vessel Networks
export function RetinalEye({ light = 0.6 }: { light?: number }) {
  const ref = useHiDpiCanvas();
  useEffect(() => {
    const canvas = ref.current!;
    const ctx = canvas.getContext('2d')!;
    let raf = 0;
    let t = 0;

    function render() {
      const W = Math.max(260, canvas.clientWidth);
      const H = Math.max(200, canvas.clientHeight);
      ctx.clearRect(0, 0, W, H);

      const cx = W / 2,
        cy = H / 2;
      const eyeR = Math.min(W, H) * 0.34;

      // sclera
      ctx.beginPath();
      ctx.arc(cx, cy, eyeR, 0, Math.PI * 2);
      ctx.fillStyle = '#fbfbfd';
      ctx.fill();
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = 'rgba(0, 122, 255, 0.1)';
      ctx.stroke();

      // retinal vessels (branching patterns)
      ctx.strokeStyle = 'rgba(0, 122, 255, 0.15)';
      ctx.lineWidth = 2;
      for (let i = 0; i < 8; i++) {
        ctx.beginPath();
        const angle = (i / 8) * Math.PI * 2;
        const startX = cx + Math.cos(angle) * eyeR * 0.3;
        const startY = cy + Math.sin(angle) * eyeR * 0.3;
        ctx.moveTo(startX, startY);

        // Create branching vessel pattern
        for (let j = 0; j < 3; j++) {
          const branchAngle = angle + (Math.random() - 0.5) * 0.5;
          const endX = cx + Math.cos(branchAngle) * eyeR * (0.6 + j * 0.1);
          const endY = cy + Math.sin(branchAngle) * eyeR * (0.6 + j * 0.1);
          ctx.lineTo(endX, endY);
        }
        ctx.stroke();
      }

      // iris with fractal patterns
      const irisR = eyeR * 0.48;
      const pupilMin = eyeR * 0.12;
      const pupilMax = eyeR * 0.28;
      const pupilR = clamp(
        pupilMax * (1 - light) + pupilMin * light,
        pupilMin,
        pupilMax
      );

      ctx.save();
      ctx.translate(cx, cy);
      const rings = 60;
      for (let r = 0; r < rings; r++) {
        const radius = pupilR + ((irisR - pupilR) * r) / (rings - 1);
        ctx.beginPath();
        const segments = Math.max(80, Math.floor(30 + radius * 0.12));
        for (let s = 0; s <= segments; s++) {
          const a = (s / segments) * Math.PI * 2;
          const noiseVal = perlin(
            Math.cos(a) * 0.7 + r * 0.06 + t * 0.02,
            Math.sin(a) * 0.7 + r * 0.06
          );
          const spoke = 1 - Math.pow(Math.abs(Math.sin(a * 3 + t * 0.3)), 0.9);
          const rOffset = (noiseVal - 0.5) * 8 * (1 - r / rings) * spoke;
          const x = Math.cos(a) * (radius + rOffset);
          const y = Math.sin(a) * (radius + rOffset);
          if (s === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.closePath();
        const alpha = 0.08 * (1 - r / rings);
        ctx.fillStyle = `rgba(29, 29, 31, ${alpha})`;
        ctx.fill();
      }

      // pupil
      ctx.beginPath();
      ctx.arc(0, 0, pupilR, 0, Math.PI * 2);
      ctx.fillStyle = '#000';
      ctx.fill();

      // corneal highlight
      const hlx = -eyeR * 0.22,
        hly = -eyeR * 0.22;
      const highlightR = eyeR * 0.12;
      const hg = ctx.createRadialGradient(hlx, hly, 2, hlx, hly, highlightR);
      hg.addColorStop(0, 'rgba(255,255,255,0.95)');
      hg.addColorStop(1, 'rgba(255,255,255,0)');
      ctx.globalCompositeOperation = 'lighter';
      ctx.fillStyle = hg;
      ctx.beginPath();
      ctx.arc(hlx, hly, highlightR, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalCompositeOperation = 'source-over';

      ctx.restore();

      t += 0.016;
      raf = requestAnimationFrame(render);
    }

    raf = requestAnimationFrame(render);
    return () => cancelAnimationFrame(raf);
  }, [ref, light]);

  return <canvas ref={ref} className="absolute inset-0 h-full w-full" />;
}

// Motor Assessment - Hand Kinematics with DH Parameters
export function HandKinematics() {
  const ref = useHiDpiCanvas();
  useEffect(() => {
    const canvas = ref.current!;
    const ctx = canvas.getContext('2d')!;
    let raf = 0;
    let t = 0;

    // Denavit-Hartenberg forward kinematics for finger joints
    function forwardKinematics(theta1: number, theta2: number, theta3: number) {
      const L1 = 45,
        L2 = 35,
        L3 = 25; // link lengths
      const pts = [{ x: 0, y: 0 }]; // base

      // MCP joint
      pts.push({
        x: L1 * Math.cos(theta1),
        y: L1 * Math.sin(theta1),
      });

      // PIP joint
      pts.push({
        x: pts[1]!.x + L2 * Math.cos(theta1 + theta2),
        y: pts[1]!.y + L2 * Math.sin(theta1 + theta2),
      });

      // DIP joint
      pts.push({
        x: pts[2]!.x + L3 * Math.cos(theta1 + theta2 + theta3),
        y: pts[2]!.y + L3 * Math.sin(theta1 + theta2 + theta3),
      });

      return pts;
    }

    function render() {
      const W = Math.max(480, canvas.clientWidth);
      const H = Math.max(320, canvas.clientHeight);
      ctx.clearRect(0, 0, W, H);

      const baseX = W * 0.3,
        baseY = H * 0.7;

      // Animate finger movement (tapping motion)
      const tapPhase = Math.sin(t * 2) * 0.5 + 0.5;
      const theta1 = -0.3 + tapPhase * 0.8;
      const theta2 = -0.2 + tapPhase * 0.6;
      const theta3 = -0.1 + tapPhase * 0.4;

      const pts = forwardKinematics(theta1, theta2, theta3);

      // Draw bones
      function drawBone(p1: any, p2: any, width: number) {
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.strokeStyle = 'rgba(0, 122, 255, 0.6)';
        ctx.lineWidth = width;
        ctx.stroke();
      }

      const p0 = { x: baseX, y: baseY };
      const p1 = { x: baseX + pts[1]!.x, y: baseY + pts[1]!.y };
      const p2 = { x: baseX + pts[2]!.x, y: baseY + pts[2]!.y };
      const p3 = { x: baseX + pts[3]!.x, y: baseY + pts[3]!.y };

      drawBone(p0, p1, 18);
      drawBone(p1, p2, 14);
      drawBone(p2, p3, 10);

      // Draw joints
      function drawJoint(p: any) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 10, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.fill();
        ctx.strokeStyle = 'rgba(0, 122, 255, 0.4)';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
      drawJoint(p0);
      drawJoint(p1);
      drawJoint(p2);

      // Draw tendons as bezier curves
      ctx.beginPath();
      ctx.moveTo(baseX - 10, baseY - 18);
      ctx.quadraticCurveTo(p1.x, p1.y - 40, p3.x + 6, p3.y - 8);
      ctx.strokeStyle = 'rgba(0, 122, 255, 0.3)';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Fingertip pad
      ctx.beginPath();
      ctx.arc(p3.x, p3.y, 6, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(0, 122, 255, 0.2)';
      ctx.fill();

      // Tremor visualization (small oscillations)
      const tremor = Math.sin(t * 15) * 2;
      ctx.beginPath();
      ctx.arc(p3.x + tremor, p3.y, 2, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255, 59, 48, 0.6)';
      ctx.fill();

      t += 0.02;
      raf = requestAnimationFrame(render);
    }

    raf = requestAnimationFrame(render);
    return () => cancelAnimationFrame(raf);
  }, [ref]);

  return <canvas ref={ref} className="absolute inset-0 h-full w-full" />;
}

// Cognitive Evaluation - Brain with Neural Activity
export function BrainNeural() {
  const ref = useHiDpiCanvas();
  useEffect(() => {
    const canvas = ref.current!;
    const ctx = canvas.getContext('2d')!;
    let raf = 0;
    let t = 0;

    function render() {
      const W = Math.max(480, canvas.clientWidth);
      const H = Math.max(360, canvas.clientHeight);
      ctx.clearRect(0, 0, W, H);

      const cx = W / 2,
        cy = H / 2;
      const rx = Math.min(W, H) * 0.46,
        ry = Math.min(W, H) * 0.32;

      // Outer hemisphere
      ctx.beginPath();
      ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
      ctx.fillStyle = '#fbfbfd';
      ctx.fill();
      ctx.strokeStyle = 'rgba(0, 122, 255, 0.1)';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Gyri layers using Perlin noise
      const layers = 28;
      for (let L = 0; L < layers; L++) {
        const ratio = L / (layers - 1);
        const baseR = rx * 0.6 + rx * 0.36 * (1 - ratio);
        ctx.beginPath();
        const steps = 240;
        for (let s = 0; s <= steps; s++) {
          const a = (s / steps) * Math.PI * 2;
          const nx = Math.cos(a) * baseR;
          const ny = Math.sin(a) * (baseR * (ry / rx));
          const n = perlin(nx * 0.015 + t * 0.02, ny * 0.015 - t * 0.01);
          const offset = (n - 0.5) * (8 + 18 * (1 - ratio));
          const x = cx + Math.cos(a) * (baseR + offset);
          const y = cy + Math.sin(a) * (baseR * (ry / rx) + offset * 0.6);
          if (s === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.strokeStyle = `rgba(0, 122, 255, ${0.15 * (1 - ratio)})`;
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Neural activity nodes
      for (let i = 0; i < 12; i++) {
        const angle = (i / 12) * Math.PI * 2;
        const radius = rx * 0.7;
        const x = cx + Math.cos(angle) * radius;
        const y = cy + Math.sin(angle) * radius * (ry / rx);
        const activity = Math.sin(t * 3 + i) * 0.5 + 0.5;

        ctx.beginPath();
        ctx.arc(x, y, 3 + activity * 4, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(0, 122, 255, ${0.3 + activity * 0.4})`;
        ctx.fill();

        // Neural connections
        if (i < 11) {
          const nextAngle = ((i + 1) / 12) * Math.PI * 2;
          const nextX = cx + Math.cos(nextAngle) * radius;
          const nextY = cy + Math.sin(nextAngle) * radius * (ry / rx);

          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(nextX, nextY);
          ctx.strokeStyle = `rgba(0, 122, 255, ${0.1 + activity * 0.2})`;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }

      // Corpus callosum
      ctx.beginPath();
      ctx.moveTo(cx - rx * 0.18, cy);
      ctx.quadraticCurveTo(cx, cy + 10, cx + rx * 0.18, cy);
      ctx.strokeStyle = 'rgba(0, 122, 255, 0.2)';
      ctx.lineWidth = 6;
      ctx.stroke();

      t += 0.01;
      raf = requestAnimationFrame(render);
    }

    raf = requestAnimationFrame(render);
    return () => cancelAnimationFrame(raf);
  }, [ref]);

  return <canvas ref={ref} className="absolute inset-0 h-full w-full" />;
}

// NRI Fusion Engine - Data Integration Visualization
export function NRIFusion() {
  const ref = useHiDpiCanvas();
  useEffect(() => {
    const canvas = ref.current!;
    const ctx = canvas.getContext('2d')!;
    let raf = 0;
    let t = 0;

    function render() {
      const W = Math.max(480, canvas.clientWidth);
      const H = Math.max(320, canvas.clientHeight);
      ctx.clearRect(0, 0, W, H);

      const cx = W / 2,
        cy = H / 2;

      // Central fusion node
      ctx.beginPath();
      ctx.arc(cx, cy, 20, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(0, 122, 255, 0.8)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(0, 122, 255, 1)';
      ctx.lineWidth = 3;
      ctx.stroke();

      // Data streams from 4 modalities
      const modalities = [
        { name: 'Speech', angle: 0, color: 'rgba(0, 122, 255, 0.6)' },
        {
          name: 'Retinal',
          angle: Math.PI / 2,
          color: 'rgba(0, 122, 255, 0.6)',
        },
        { name: 'Motor', angle: Math.PI, color: 'rgba(0, 122, 255, 0.6)' },
        {
          name: 'Cognitive',
          angle: (3 * Math.PI) / 2,
          color: 'rgba(0, 122, 255, 0.6)',
        },
      ];

      modalities.forEach((mod, i) => {
        const radius = 80;
        const x = cx + Math.cos(mod.angle) * radius;
        const y = cy + Math.sin(mod.angle) * radius;

        // Modality node
        ctx.beginPath();
        ctx.arc(x, y, 12, 0, Math.PI * 2);
        ctx.fillStyle = mod.color;
        ctx.fill();

        // Animated data flow
        const flowPhase = (t * 2 + (i * Math.PI) / 2) % (Math.PI * 2);
        const flowProgress = (Math.sin(flowPhase) + 1) / 2;

        // Data particles flowing toward center
        for (let j = 0; j < 5; j++) {
          const particleProgress = (flowProgress + j * 0.2) % 1;
          const px = x + (cx - x) * particleProgress;
          const py = y + (cy - y) * particleProgress;

          ctx.beginPath();
          ctx.arc(px, py, 2 + Math.sin(t * 4 + j) * 1, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(0, 122, 255, ${0.4 + particleProgress * 0.4})`;
          ctx.fill();
        }

        // Connection lines
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(cx, cy);
        ctx.strokeStyle = `rgba(0, 122, 255, ${0.2 + Math.sin(t * 2 + i) * 0.1})`;
        ctx.lineWidth = 2;
        ctx.stroke();
      });

      // Fusion algorithm visualization (concentric rings)
      for (let r = 0; r < 3; r++) {
        const radius = 30 + r * 15;
        const alpha = 0.1 + Math.sin(t * 1.5 + r) * 0.05;

        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(0, 122, 255, ${alpha})`;
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      t += 0.02;
      raf = requestAnimationFrame(render);
    }

    raf = requestAnimationFrame(render);
    return () => cancelAnimationFrame(raf);
  }, [ref]);

  return <canvas ref={ref} className="absolute inset-0 h-full w-full" />;
}

// Multi-Modal Assessment - Integrated Network Visualization
export function MultiModalNetwork() {
  const ref = useHiDpiCanvas();
  useEffect(() => {
    const canvas = ref.current!;
    const ctx = canvas.getContext('2d')!;
    let raf = 0;
    let t = 0;

    function render() {
      const W = Math.max(480, canvas.clientWidth);
      const H = Math.max(360, canvas.clientHeight);
      ctx.clearRect(0, 0, W, H);

      const cx = W / 2,
        cy = H / 2;

      // Network nodes representing different assessment types
      const nodes = [
        { x: cx - 80, y: cy - 60, label: 'Speech', size: 15 },
        { x: cx + 80, y: cy - 60, label: 'Retinal', size: 15 },
        { x: cx - 80, y: cy + 60, label: 'Motor', size: 15 },
        { x: cx + 80, y: cy + 60, label: 'Cognitive', size: 15 },
        { x: cx, y: cy, label: 'NRI', size: 20 },
      ];

      // Draw connections between all nodes
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const activity = Math.sin(t * 2 + i + j) * 0.5 + 0.5;

          ctx.beginPath();
          ctx.moveTo(nodes[i]!.x, nodes[i]!.y);
          ctx.lineTo(nodes[j]!.x, nodes[j]!.y);
          ctx.strokeStyle = `rgba(0, 122, 255, ${0.1 + activity * 0.2})`;
          ctx.lineWidth = 1 + activity * 2;
          ctx.stroke();

          // Data packets traveling along connections
          const packetProgress = (t * 1.5 + i * j) % 1;
          const px = nodes[i]!.x + (nodes[j]!.x - nodes[i]!.x) * packetProgress;
          const py = nodes[i]!.y + (nodes[j]!.y - nodes[i]!.y) * packetProgress;

          ctx.beginPath();
          ctx.arc(px, py, 2, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(0, 122, 255, ${0.6 + activity * 0.4})`;
          ctx.fill();
        }
      }

      // Draw nodes
      nodes.forEach((node, i) => {
        const pulse = Math.sin(t * 3 + i) * 0.3 + 0.7;

        ctx.beginPath();
        ctx.arc(node.x, node.y, node.size * pulse, 0, Math.PI * 2);
        ctx.fillStyle =
          i === 4 ? 'rgba(0, 122, 255, 0.8)' : 'rgba(0, 122, 255, 0.6)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(0, 122, 255, 1)';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Node activity rings
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.size * (1 + pulse * 0.5), 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(0, 122, 255, ${0.2 * pulse})`;
        ctx.lineWidth = 1;
        ctx.stroke();
      });

      // Integration waves emanating from center
      for (let w = 0; w < 2; w++) {
        const waveRadius = (t * 50 + w * 50) % 150;
        const alpha = 1 - waveRadius / 150;

        if (alpha > 0) {
          ctx.beginPath();
          ctx.arc(cx, cy, waveRadius, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(0, 122, 255, ${alpha * 0.3})`;
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }

      t += 0.016;
      raf = requestAnimationFrame(render);
    }

    raf = requestAnimationFrame(render);
    return () => cancelAnimationFrame(raf);
  }, [ref]);

  return <canvas ref={ref} className="absolute inset-0 h-full w-full" />;
}
