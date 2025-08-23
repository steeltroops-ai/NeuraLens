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

// Motor Assessment - Enhanced Hand Kinematics with Anatomical Features
export function HandKinematics() {
  const ref = useHiDpiCanvas();
  useEffect(() => {
    const canvas = ref.current!;
    const ctx = canvas.getContext('2d')!;
    let raf = 0;
    let t = 0;
    const trailPoints: Array<{ x: number; y: number; time: number }> = [];

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

    // Draw anatomical bone structure
    function drawBone(p1: any, p2: any, width: number) {
      // Bone shaft with anatomical shape
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.strokeStyle = '#E8E8E8';
      ctx.lineWidth = width;
      ctx.lineCap = 'round';
      ctx.stroke();

      // Bone highlights for 3D effect
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = width * 0.3;
      ctx.stroke();
    }

    // Draw anatomical joint markers
    function drawJoint(p: any, radius: number, isActive: boolean = false) {
      // Joint capsule
      ctx.beginPath();
      ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = isActive ? '#007AFF' : '#F0F0F0';
      ctx.fill();
      ctx.strokeStyle = '#D0D0D0';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Joint center
      ctx.beginPath();
      ctx.arc(p.x, p.y, radius * 0.4, 0, Math.PI * 2);
      ctx.fillStyle = isActive ? '#FFFFFF' : '#C0C0C0';
      ctx.fill();
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

      const p0 = { x: baseX, y: baseY };
      const p1 = { x: baseX + pts[1]!.x, y: baseY + pts[1]!.y };
      const p2 = { x: baseX + pts[2]!.x, y: baseY + pts[2]!.y };
      const p3 = { x: baseX + pts[3]!.x, y: baseY + pts[3]!.y };

      // Draw palm base (anatomical structure)
      ctx.beginPath();
      ctx.ellipse(baseX - 15, baseY + 10, 25, 15, 0, 0, Math.PI * 2);
      ctx.fillStyle = '#F5F5F5';
      ctx.fill();
      ctx.strokeStyle = '#D0D0D0';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Draw enhanced anatomical bones
      drawBone(p0, p1, 18);
      drawBone(p1, p2, 14);
      drawBone(p2, p3, 10);

      // Draw enhanced anatomical joints with activity indicators
      const isActive = tapPhase > 0.7;
      drawJoint(p0, 12, isActive);
      drawJoint(p1, 10, isActive);
      drawJoint(p2, 8, isActive);

      // Draw tendon pathways (curved anatomical connections)
      ctx.beginPath();
      ctx.moveTo(baseX - 10, baseY - 18);
      ctx.quadraticCurveTo(p1.x - 5, p1.y - 25, p2.x, p2.y - 15);
      ctx.quadraticCurveTo(p2.x + 3, p2.y - 8, p3.x + 6, p3.y - 8);
      ctx.strokeStyle = 'rgba(255, 140, 0, 0.6)';
      ctx.lineWidth = 3;
      ctx.stroke();

      // Enhanced fingertip pad with anatomical detail
      ctx.beginPath();
      ctx.arc(p3.x, p3.y, 8, 0, Math.PI * 2);
      ctx.fillStyle = '#FFE4E1';
      ctx.fill();
      ctx.strokeStyle = '#D0D0D0';
      ctx.lineWidth = 1;
      ctx.stroke();

      // Fingerprint pattern (subtle detail)
      for (let i = 0; i < 3; i++) {
        ctx.beginPath();
        ctx.arc(p3.x, p3.y, 3 + i * 1.5, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(200, 200, 200, ${0.3 - i * 0.1})`;
        ctx.lineWidth = 0.5;
        ctx.stroke();
      }

      // Enhanced tremor visualization with multiple oscillation points
      const tremor = Math.sin(t * 15) * 2;
      const microTremor = Math.sin(t * 25) * 0.8;

      // Primary tremor indicator
      ctx.beginPath();
      ctx.arc(p3.x + tremor, p3.y + microTremor, 3, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255, 59, 48, 0.8)';
      ctx.fill();

      // Secondary tremor indicators
      ctx.beginPath();
      ctx.arc(p2.x + tremor * 0.6, p2.y + microTremor * 0.4, 2, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255, 59, 48, 0.5)';
      ctx.fill();

      // Movement trail effect
      trailPoints.push({ x: p3.x, y: p3.y, time: t });

      // Remove old trail points
      while (trailPoints.length > 0 && t - trailPoints[0]!.time > 2) {
        trailPoints.shift();
      }

      // Draw movement trail
      for (let i = 0; i < trailPoints.length - 1; i++) {
        const point = trailPoints[i]!;
        const alpha = (t - point.time) / 2;
        ctx.beginPath();
        ctx.arc(point.x, point.y, 1, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(0, 122, 255, ${0.4 * (1 - alpha)})`;
        ctx.fill();
      }

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

      // Enhanced neural activity nodes with specialized brain regions
      const brainRegions = [
        { name: 'Frontal', color: '#FF6B6B', cognitive: true },
        { name: 'Parietal', color: '#4ECDC4', attention: true },
        { name: 'Temporal', color: '#45B7D1', memory: true },
        { name: 'Occipital', color: '#96CEB4', visual: true },
        { name: 'Motor', color: '#FFEAA7', motor: true },
        { name: 'Sensory', color: '#DDA0DD', sensory: true },
      ];

      for (let i = 0; i < 12; i++) {
        const angle = (i / 12) * Math.PI * 2;
        const radius = rx * 0.7;
        const x = cx + Math.cos(angle) * radius;
        const y = cy + Math.sin(angle) * radius * (ry / rx);
        const activity = Math.sin(t * 3 + i) * 0.5 + 0.5;
        const region = brainRegions[i % brainRegions.length]!;

        // Enhanced neural nodes with region-specific colors
        ctx.beginPath();
        ctx.arc(x, y, 4 + activity * 6, 0, Math.PI * 2);
        ctx.fillStyle = region.color;
        ctx.globalAlpha = 0.4 + activity * 0.6;
        ctx.fill();
        ctx.globalAlpha = 1;

        // Node outline
        ctx.beginPath();
        ctx.arc(x, y, 4 + activity * 6, 0, Math.PI * 2);
        ctx.strokeStyle = region.color;
        ctx.lineWidth = 2;
        ctx.stroke();

        // Synaptic firing effects (bright flashes)
        if (activity > 0.8) {
          ctx.beginPath();
          ctx.arc(x, y, 2, 0, Math.PI * 2);
          ctx.fillStyle = '#FFFFFF';
          ctx.fill();

          // Firing burst effect
          for (let burst = 0; burst < 4; burst++) {
            const burstAngle = (burst / 4) * Math.PI * 2;
            const burstX = x + Math.cos(burstAngle) * 8;
            const burstY = y + Math.sin(burstAngle) * 8;
            ctx.beginPath();
            ctx.arc(burstX, burstY, 1, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255, 255, 255, ${0.8 * activity})`;
            ctx.fill();
          }
        }

        // Enhanced neural pathways with branching
        for (let j = i + 1; j < 12; j++) {
          const nextAngle = (j / 12) * Math.PI * 2;
          const nextX = cx + Math.cos(nextAngle) * radius;
          const nextY = cy + Math.sin(nextAngle) * radius * (ry / rx);
          const nextActivity = Math.sin(t * 3 + j) * 0.5 + 0.5;
          const connectionStrength = (activity + nextActivity) / 2;

          if (connectionStrength > 0.4) {
            // Main pathway
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(nextX, nextY);
            ctx.strokeStyle = `rgba(0, 122, 255, ${connectionStrength * 0.3})`;
            ctx.lineWidth = 1 + connectionStrength * 2;
            ctx.stroke();

            // Branching pathways
            const midX = (x + nextX) / 2;
            const midY = (y + nextY) / 2;
            const branchAngle = Math.atan2(nextY - y, nextX - x) + Math.PI / 4;
            const branchX = midX + Math.cos(branchAngle) * 15;
            const branchY = midY + Math.sin(branchAngle) * 15;

            ctx.beginPath();
            ctx.moveTo(midX, midY);
            ctx.lineTo(branchX, branchY);
            ctx.strokeStyle = `rgba(0, 122, 255, ${connectionStrength * 0.2})`;
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }
      }

      // Enhanced corpus callosum with information transfer
      ctx.beginPath();
      ctx.moveTo(cx - rx * 0.18, cy);
      ctx.quadraticCurveTo(cx, cy + 10, cx + rx * 0.18, cy);
      ctx.strokeStyle = 'rgba(0, 122, 255, 0.3)';
      ctx.lineWidth = 8;
      ctx.stroke();

      // Information transfer particles across corpus callosum
      const transferPhase = (t * 2) % 1;
      const transferX = cx - rx * 0.18 + rx * 0.36 * transferPhase;
      const transferY = cy + 10 * Math.sin(Math.PI * transferPhase);
      ctx.beginPath();
      ctx.arc(transferX, transferY, 3, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255, 215, 0, 0.8)';
      ctx.fill();

      // Memory formation clusters (hippocampus region)
      const memoryX = cx - rx * 0.3;
      const memoryY = cy + ry * 0.4;
      const memoryActivity = Math.sin(t * 1.5) * 0.5 + 0.5;

      for (let m = 0; m < 6; m++) {
        const clusterAngle = (m / 6) * Math.PI * 2;
        const clusterRadius = 12 + memoryActivity * 8;
        const clusterX = memoryX + Math.cos(clusterAngle) * clusterRadius;
        const clusterY = memoryY + Math.sin(clusterAngle) * clusterRadius;

        ctx.beginPath();
        ctx.arc(clusterX, clusterY, 2 + memoryActivity * 3, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 140, 0, ${0.4 + memoryActivity * 0.4})`;
        ctx.fill();

        // Memory consolidation connections
        if (memoryActivity > 0.6) {
          ctx.beginPath();
          ctx.moveTo(memoryX, memoryY);
          ctx.lineTo(clusterX, clusterY);
          ctx.strokeStyle = `rgba(255, 140, 0, ${memoryActivity * 0.3})`;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }

      // Attention network (frontal-parietal connections)
      const frontalX = cx;
      const frontalY = cy - ry * 0.6;
      const parietalX = cx;
      const parietalY = cy + ry * 0.3;
      const attentionFlow = Math.sin(t * 4) * 0.5 + 0.5;

      // Flowing particles showing attention direction
      for (let a = 0; a < 3; a++) {
        const flowProgress = (attentionFlow + a * 0.33) % 1;
        const flowX = frontalX + (parietalX - frontalX) * flowProgress;
        const flowY = frontalY + (parietalY - frontalY) * flowProgress;

        ctx.beginPath();
        ctx.arc(flowX, flowY, 2, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(138, 43, 226, ${0.6 + attentionFlow * 0.4})`;
        ctx.fill();
      }

      // Cognitive load visualization through overall brightness modulation
      const cognitiveLoad = Math.sin(t * 0.8) * 0.3 + 0.7;
      ctx.globalAlpha = cognitiveLoad;

      // Brain activity waves
      for (let w = 0; w < 2; w++) {
        const waveRadius = (t * 30 + w * 40) % 120;
        const waveAlpha = 1 - waveRadius / 120;

        if (waveAlpha > 0) {
          ctx.beginPath();
          ctx.arc(cx, cy, waveRadius, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(138, 43, 226, ${waveAlpha * 0.2 * cognitiveLoad})`;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }

      ctx.globalAlpha = 1;

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

      // Enhanced central fusion node with processing visualization
      const fusionActivity = Math.sin(t * 3) * 0.3 + 0.7;
      ctx.beginPath();
      ctx.arc(cx, cy, 25 * fusionActivity, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255, 215, 0, 0.8)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(255, 215, 0, 1)';
      ctx.lineWidth = 4;
      ctx.stroke();

      // Fusion algorithm visualization (mathematical processing)
      const symbols = ['∑', '∫', '∆', '∇'];
      for (let s = 0; s < symbols.length; s++) {
        const symbolAngle = (s / symbols.length) * Math.PI * 2 + t;
        const symbolRadius = 15;
        const symbolX = cx + Math.cos(symbolAngle) * symbolRadius;
        const symbolY = cy + Math.sin(symbolAngle) * symbolRadius;

        ctx.fillStyle = `rgba(255, 255, 255, ${0.6 + Math.sin(t * 4 + s) * 0.3})`;
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(symbols[s]!, symbolX, symbolY + 5);
      }

      // Enhanced modalities with distinct visual representations
      const modalities = [
        {
          name: 'Speech',
          angle: 0,
          color: '#4A90E2',
          icon: '🎤',
          symbol: '♪',
        },
        {
          name: 'Retinal',
          angle: Math.PI / 2,
          color: '#50C878',
          icon: '👁️',
          symbol: '◉',
        },
        {
          name: 'Motor',
          angle: Math.PI,
          color: '#FF8C42',
          icon: '✋',
          symbol: '⚡',
        },
        {
          name: 'Cognitive',
          angle: (3 * Math.PI) / 2,
          color: '#9B59B6',
          icon: '🧠',
          symbol: '⚙',
        },
      ];

      modalities.forEach((mod, i) => {
        const radius = 80;
        const x = cx + Math.cos(mod.angle) * radius;
        const y = cy + Math.sin(mod.angle) * radius;
        const activity = Math.sin(t * 2 + i) * 0.5 + 0.5;

        // Enhanced modality node with distinct colors
        ctx.beginPath();
        ctx.arc(x, y, 15 + activity * 5, 0, Math.PI * 2);
        ctx.fillStyle = mod.color;
        ctx.fill();
        ctx.strokeStyle = mod.color;
        ctx.lineWidth = 3;
        ctx.stroke();

        // Modality symbol/icon representation
        ctx.fillStyle = '#FFFFFF';
        ctx.font = '16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(mod.symbol, x, y + 5);

        // Processing stage (intermediate node)
        const stageRadius = 50;
        const stageX = cx + Math.cos(mod.angle) * stageRadius;
        const stageY = cy + Math.sin(mod.angle) * stageRadius;

        ctx.beginPath();
        ctx.arc(stageX, stageY, 6, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 255, 255, ${0.6 + activity * 0.4})`;
        ctx.fill();
        ctx.strokeStyle = mod.color;
        ctx.lineWidth = 2;
        ctx.stroke();

        // Enhanced data flow with confidence indicators
        const flowPhase = (t * 2 + (i * Math.PI) / 2) % (Math.PI * 2);
        const flowProgress = (Math.sin(flowPhase) + 1) / 2;
        const confidence = 0.7 + Math.sin(t * 1.5 + i) * 0.3;

        // Data particles with varying opacity based on confidence
        for (let j = 0; j < 5; j++) {
          const particleProgress = (flowProgress + j * 0.2) % 1;
          const px = x + (cx - x) * particleProgress;
          const py = y + (cy - y) * particleProgress;

          ctx.beginPath();
          ctx.arc(px, py, 2 + Math.sin(t * 4 + j) * 1, 0, Math.PI * 2);
          const alpha = (0.4 + particleProgress * 0.4) * confidence;
          ctx.fillStyle = `${mod.color.slice(0, -1)}, ${alpha})`;
          ctx.fill();
        }

        // Connection lines with varying thickness based on correlation
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(stageX, stageY);
        ctx.lineTo(cx, cy);
        ctx.strokeStyle = `${mod.color.slice(0, -1)}, ${0.3 + activity * 0.4})`;
        ctx.lineWidth = 2 + activity * 3;
        ctx.stroke();

        // Correlation analysis arcs between modalities
        for (let k = i + 1; k < modalities.length; k++) {
          const otherMod = modalities[k]!;
          const otherX = cx + Math.cos(otherMod.angle) * radius;
          const otherY = cy + Math.sin(otherMod.angle) * radius;
          const correlation = Math.sin(t * 1.2 + i + k) * 0.5 + 0.5;

          if (correlation > 0.6) {
            // Draw correlation arc
            const midX = (x + otherX) / 2;
            const midY = (y + otherY) / 2;
            const arcRadius =
              Math.sqrt((x - otherX) ** 2 + (y - otherY) ** 2) / 3;

            ctx.beginPath();
            ctx.arc(midX, midY, arcRadius, 0, Math.PI * 2);
            ctx.strokeStyle = `rgba(255, 215, 0, ${correlation * 0.3})`;
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }
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

      // Enhanced network nodes with distinct colors and icons
      const nodes = [
        {
          x: cx - 80,
          y: cy - 60,
          label: 'Speech',
          size: 15,
          color: '#4A90E2',
          icon: '♪',
          symbol: '🎤',
        },
        {
          x: cx + 80,
          y: cy - 60,
          label: 'Retinal',
          size: 15,
          color: '#50C878',
          icon: '◉',
          symbol: '👁️',
        },
        {
          x: cx - 80,
          y: cy + 60,
          label: 'Motor',
          size: 15,
          color: '#FF8C42',
          icon: '⚡',
          symbol: '✋',
        },
        {
          x: cx + 80,
          y: cy + 60,
          label: 'Cognitive',
          size: 15,
          color: '#9B59B6',
          icon: '⚙',
          symbol: '🧠',
        },
        {
          x: cx,
          y: cy,
          label: 'NRI',
          size: 20,
          color: '#FFD700',
          icon: '⚡',
          symbol: '⚡',
        },
      ];

      // Enhanced connections with network strength indicators
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const activity = Math.sin(t * 2 + i + j) * 0.5 + 0.5;
          const nodeI = nodes[i]!;
          const nodeJ = nodes[j]!;
          const connectionStrength = 0.3 + activity * 0.7;

          // Enhanced connection lines with varying thickness
          ctx.beginPath();
          ctx.moveTo(nodeI.x, nodeI.y);
          ctx.lineTo(nodeJ.x, nodeJ.y);
          ctx.strokeStyle = `rgba(100, 100, 100, ${0.2 + activity * 0.3})`;
          ctx.lineWidth = 1 + connectionStrength * 3;
          ctx.stroke();

          // Colored data packets with modality-specific colors
          const packetProgress = (t * 1.5 + i * j) % 1;
          const px = nodeI.x + (nodeJ.x - nodeI.x) * packetProgress;
          const py = nodeI.y + (nodeJ.y - nodeI.y) * packetProgress;

          // Use source node color for data packet
          const packetColor = i === 4 ? nodeJ.color : nodeI.color;

          ctx.beginPath();
          ctx.rect(px - 2, py - 2, 4, 4);
          ctx.fillStyle = `${packetColor}CC`;
          ctx.fill();
          ctx.strokeStyle = packetColor;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }

      // Enhanced nodes with distinct colors and icons
      nodes.forEach((node, i) => {
        const pulse = Math.sin(t * 3 + i) * 0.3 + 0.7;

        // Main node with modality-specific color
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.size * pulse, 0, Math.PI * 2);
        ctx.fillStyle = node.color;
        ctx.fill();
        ctx.strokeStyle = node.color;
        ctx.lineWidth = 3;
        ctx.stroke();

        // Node icon/symbol
        ctx.fillStyle = '#FFFFFF';
        ctx.font = `${node.size}px Arial`;
        ctx.textAlign = 'center';
        ctx.fillText(node.icon, node.x, node.y + node.size * 0.3);

        // Enhanced activity rings with modality colors
        ctx.beginPath();
        ctx.arc(node.x, node.y, node.size * (1 + pulse * 0.5), 0, Math.PI * 2);
        ctx.strokeStyle = `${node.color}66`;
        ctx.lineWidth = 2;
        ctx.stroke();

        // Outer activity ring
        ctx.beginPath();
        ctx.arc(
          node.x,
          node.y,
          node.size * (1.5 + pulse * 0.3),
          0,
          Math.PI * 2
        );
        ctx.strokeStyle = `${node.color}33`;
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
