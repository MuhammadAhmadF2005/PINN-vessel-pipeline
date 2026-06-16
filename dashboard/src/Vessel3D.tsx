import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import gsap from 'gsap';

interface Vessel3DProps {
  temperature: number;
  pressure: number;
  isAnomaly: boolean;
  activeAlerts: string[];
}

export const Vessel3D: React.FC<Vessel3DProps> = ({
  temperature,
  pressure,
  isAnomaly,
  activeAlerts,
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const sealTween = useRef<gsap.core.Tween | null>(null);
  const heaterTween = useRef<gsap.core.Tween | null>(null);
  const blockageTween = useRef<gsap.core.Tween | null>(null);

  // Keep alert statuses in refs to read them in the animation loop and trigger tweens
  const propsRef = useRef({ temperature, pressure, isAnomaly, activeAlerts });
  useEffect(() => {
    propsRef.current = { temperature, pressure, isAnomaly, activeAlerts };
  }, [temperature, pressure, isAnomaly, activeAlerts]);

  useEffect(() => {
    if (!mountRef.current) return;

    // 1. Scene, Camera, Renderer Setup
    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight || 450;
    const scene = new THREE.Scene();
    
    // Cyber-dark scene background
    scene.background = null; // transparent canvas to inherit glassmorphic panel styling

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.set(0, 3, 9);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mountRef.current.appendChild(renderer.domElement);

    // 2. Lighting Setup (Neon & Cyber Aesthetics)
    const ambientLight = new THREE.AmbientLight(0x0e1726, 1.5);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0x0ea5e9, 2.0); // Cyan light
    dirLight1.position.set(5, 5, 5);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xd946ef, 1.5); // Purple backlight
    dirLight2.position.set(-5, -3, -5);
    scene.add(dirLight2);

    const pointLight = new THREE.PointLight(0xffffff, 1.0, 15);
    pointLight.position.set(0, 0, 3);
    scene.add(pointLight);

    // 3. Orbit Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxDistance = 15;
    controls.minDistance = 4;
    controls.enablePan = false;

    // 4. Vessel Group setup
    const vesselGroup = new THREE.Group();
    scene.add(vesselGroup);

    // 5. Vessel Geometry (Capsule/Cylinder shape)
    // Translucent glass outer wall
    const wallGeo = new THREE.CylinderGeometry(1.5, 1.5, 4.0, 32, 1, true);
    const wallMat = new THREE.MeshPhysicalMaterial({
      color: 0x0ea5e9,
      transparent: true,
      opacity: 0.35,
      roughness: 0.1,
      metalness: 0.1,
      clearcoat: 1.0,
      clearcoatRoughness: 0.1,
      side: THREE.DoubleSide,
      depthWrite: false, // Prevents particle rendering sorting artifacts
    });
    const wallMesh = new THREE.Mesh(wallGeo, wallMat);
    vesselGroup.add(wallMesh);

    // Top Dome
    const topDomeGeo = new THREE.SphereGeometry(1.5, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2);
    const domeMat = wallMat.clone();
    const topDomeMesh = new THREE.Mesh(topDomeGeo, domeMat);
    topDomeMesh.position.y = 2.0;
    vesselGroup.add(topDomeMesh);

    // Bottom Dome
    const bottomDomeGeo = new THREE.SphereGeometry(1.5, 32, 16, 0, Math.PI * 2, Math.PI / 2, Math.PI / 2);
    const bottomDomeMesh = new THREE.Mesh(bottomDomeGeo, domeMat);
    bottomDomeMesh.position.y = -2.0;
    vesselGroup.add(bottomDomeMesh);

    // 6. Highlight Zones (Seal, Heater, Inlet)
    // A. Seal Ring (Top Junction)
    const sealRingGeo = new THREE.TorusGeometry(1.5, 0.06, 16, 100);
    const sealRingMat = new THREE.MeshStandardMaterial({
      color: 0x334155,
      metalness: 0.8,
      roughness: 0.2,
      emissive: new THREE.Color(0x000000),
    });
    const sealRingMesh = new THREE.Mesh(sealRingGeo, sealRingMat);
    sealRingMesh.position.y = 2.0;
    sealRingMesh.rotation.x = Math.PI / 2;
    vesselGroup.add(sealRingMesh);

    // B. Inlet Pipe (Sticking out of Top Dome)
    const inletGeo = new THREE.CylinderGeometry(0.25, 0.25, 0.8, 16);
    const inletMat = new THREE.MeshStandardMaterial({
      color: 0x334155,
      metalness: 0.8,
      roughness: 0.2,
      emissive: new THREE.Color(0x000000),
    });
    const inletMesh = new THREE.Mesh(inletGeo, inletMat);
    inletMesh.position.set(0, 3.1, 0);
    vesselGroup.add(inletMesh);

    // C. Heater Coil (Wrapped around lower cylinder)
    const curvePoints: THREE.Vector3[] = [];
    const turns = 5;
    const pointsPerTurn = 40;
    const totalPoints = turns * pointsPerTurn;
    for (let i = 0; i <= totalPoints; i++) {
      const theta = (i / pointsPerTurn) * Math.PI * 2;
      const x = Math.cos(theta) * 1.6; // slightly wider than 1.5 radius wall
      const z = Math.sin(theta) * 1.6;
      const y = -1.6 + (i / totalPoints) * 2.8; // wrapped from y=-1.6 to y=1.2
      curvePoints.push(new THREE.Vector3(x, y, z));
    }
    const helixCurve = new THREE.CatmullRomCurve3(curvePoints);
    const heaterGeo = new THREE.TubeGeometry(helixCurve, 120, 0.05, 8, false);
    const heaterMat = new THREE.MeshStandardMaterial({
      color: 0x334155,
      metalness: 0.9,
      roughness: 0.1,
      emissive: new THREE.Color(0x000000),
    });
    const heaterMesh = new THREE.Mesh(heaterGeo, heaterMat);
    vesselGroup.add(heaterMesh);

    // 7. Internal Pressure Particle System
    const particleCount = 250;
    const particleGeo = new THREE.BufferGeometry();
    const particlePositions = new Float32Array(particleCount * 3);
    const velocities: { ySpeed: number; angleSpeed: number; radius: number; angle: number }[] = [];

    for (let i = 0; i < particleCount; i++) {
      // Pick random height, radius, and angle
      const y = (Math.random() - 0.5) * 4.0; // from -2.0 to 2.0
      const r = Math.random() * 1.35; // less than 1.5
      const angle = Math.random() * Math.PI * 2;

      particlePositions[i * 3] = Math.cos(angle) * r;
      particlePositions[i * 3 + 1] = y;
      particlePositions[i * 3 + 2] = Math.sin(angle) * r;

      velocities.push({
        ySpeed: (0.2 + Math.random() * 0.8) * 0.02,
        angleSpeed: (Math.random() - 0.5) * 0.05,
        radius: r,
        angle: angle,
      });
    }

    particleGeo.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));
    
    // Canvas texture for glowing round particles
    const createParticleTexture = () => {
      const canvas = document.createElement('canvas');
      canvas.width = 16;
      canvas.height = 16;
      const ctx = canvas.getContext('2d')!;
      const grad = ctx.createRadialGradient(8, 8, 0, 8, 8, 8);
      grad.addColorStop(0, 'rgba(255,255,255,1)');
      grad.addColorStop(0.3, 'rgba(14,165,233,0.8)');
      grad.addColorStop(1, 'rgba(14,165,233,0)');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, 16, 16);
      return new THREE.CanvasTexture(canvas);
    };

    const particleMat = new THREE.PointsMaterial({
      size: 0.15,
      map: createParticleTexture(),
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    const particles = new THREE.Points(particleGeo, particleMat);
    vesselGroup.add(particles);

    // 8. Animation & Render Loop
    let animationFrameId: number;
    
    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);

      // A. Ambient rotation
      vesselGroup.rotation.y += 0.002;

      // B. Dynamic Temperature Mapping on Outer Wall
      // Normal temperature is around 310 Kelvin. Scale: 300 to 340
      const curTemp = propsRef.current.temperature || 310.0;
      const tempNormalized = Math.min(Math.max((curTemp - 300.0) / 40.0, 0), 1);
      
      const coolColor = new THREE.Color(0x0ea5e9); // Neon cyan
      const hotColor = new THREE.Color(0xef4444);  // Hot red
      const currentWallColor = coolColor.clone().lerp(hotColor, tempNormalized);
      
      wallMat.color.copy(currentWallColor);
      domeMat.color.copy(currentWallColor);
      
      // Let the material glow on temperature
      wallMat.emissive.copy(currentWallColor).multiplyScalar(tempNormalized * 0.25);
      domeMat.emissive.copy(currentWallColor).multiplyScalar(tempNormalized * 0.25);

      // C. Dynamic Pressure Particle Animation
      // Normal pressure is around 4.5 to 5.0. Scale: 1.0 to 6.0
      const curPres = propsRef.current.pressure || 4.5;
      const pressureNormalized = Math.min(Math.max((curPres - 1.0) / 5.0, 0.1), 2.5);
      
      // Change particle color based on pressure
      const particleCoolColor = new THREE.Color(0x38bdf8); // Light blue
      const particleHotColor = new THREE.Color(0xfb923c);  // Orange
      particleMat.color.copy(particleCoolColor.clone().lerp(particleHotColor, Math.min(pressureNormalized, 1)));
      particleMat.size = 0.1 + pressureNormalized * 0.1;

      const posAttrib = particleGeo.getAttribute('position') as THREE.BufferAttribute;
      const positionsArray = posAttrib.array as Float32Array;

      for (let i = 0; i < particleCount; i++) {
        const vel = velocities[i];
        
        // Speed updates scale with pressure
        vel.angle += vel.angleSpeed * pressureNormalized;
        let y = positionsArray[i * 3 + 1];
        y += vel.ySpeed * pressureNormalized;

        // Reset if goes out of vertical bounds
        if (y > 2.0) {
          y = -2.0;
        }

        positionsArray[i * 3] = Math.cos(vel.angle) * vel.radius;
        positionsArray[i * 3 + 1] = y;
        positionsArray[i * 3 + 2] = Math.sin(vel.angle) * vel.radius;
      }
      posAttrib.needsUpdate = true;

      // D. Alert Highlight Tweens (Driven by activeAlerts)
      const alerts = propsRef.current.activeAlerts || [];
      const hasSeal = alerts.some(msg => msg.toLowerCase().includes('seal'));
      const hasHeater = alerts.some(msg => msg.toLowerCase().includes('heater'));
      const hasBlockage = alerts.some(msg => msg.toLowerCase().includes('blockage') || msg.toLowerCase().includes('block'));

      // Seal Degradation Alert -> Pulse Seal Ring
      if (hasSeal) {
        if (!sealTween.current) {
          sealRingMat.color.setHex(0xef4444);
          sealTween.current = gsap.to(sealRingMat.emissive, {
            r: 1.0, g: 0.1, b: 0.1,
            duration: 0.35,
            repeat: -1,
            yoyo: true,
            ease: "sine.inOut"
          });
        }
      } else {
        if (sealTween.current) {
          sealTween.current.kill();
          sealTween.current = null;
          gsap.to(sealRingMat.emissive, { r: 0, g: 0, b: 0, duration: 0.3 });
          gsap.to(sealRingMat.color, { r: 0.2, g: 0.25, b: 0.33, duration: 0.3 }); // reset to slate
        }
      }

      // Heater Drift Alert -> Pulse Heater Coil
      if (hasHeater) {
        if (!heaterTween.current) {
          heaterMat.color.setHex(0xfb923c); // Orange-yellow
          heaterTween.current = gsap.to(heaterMat.emissive, {
            r: 1.0, g: 0.4, b: 0.0,
            duration: 0.3,
            repeat: -1,
            yoyo: true,
            ease: "power1.inOut"
          });
        }
      } else {
        if (heaterTween.current) {
          heaterTween.current.kill();
          heaterTween.current = null;
          gsap.to(heaterMat.emissive, { r: 0, g: 0, b: 0, duration: 0.3 });
          gsap.to(heaterMat.color, { r: 0.2, g: 0.25, b: 0.33, duration: 0.3 });
        }
      }

      // Blockage Alert -> Pulse Inlet Pipe
      if (hasBlockage) {
        if (!blockageTween.current) {
          inletMat.color.setHex(0xef4444);
          blockageTween.current = gsap.to(inletMat.emissive, {
            r: 1.0, g: 0.0, b: 0.0,
            duration: 0.25,
            repeat: -1,
            yoyo: true,
            ease: "bounce.inOut"
          });
        }
      } else {
        if (blockageTween.current) {
          blockageTween.current.kill();
          blockageTween.current = null;
          gsap.to(inletMat.emissive, { r: 0, g: 0, b: 0, duration: 0.3 });
          gsap.to(inletMat.color, { r: 0.2, g: 0.25, b: 0.33, duration: 0.3 });
        }
      }

      controls.update();
      renderer.render(scene, camera);
    };

    animate();

    // 9. Resize Handling
    const handleResize = () => {
      if (!mountRef.current) return;
      const w = mountRef.current.clientWidth;
      const h = mountRef.current.clientHeight || 450;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    // 10. Cleanups
    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', handleResize);
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
      
      // Kill tweens
      if (sealTween.current) sealTween.current.kill();
      if (heaterTween.current) heaterTween.current.kill();
      if (blockageTween.current) blockageTween.current.kill();

      // Dispose resources
      wallGeo.dispose();
      wallMat.dispose();
      topDomeGeo.dispose();
      bottomDomeGeo.dispose();
      domeMat.dispose();
      sealRingGeo.dispose();
      sealRingMat.dispose();
      inletGeo.dispose();
      inletMat.dispose();
      heaterGeo.dispose();
      heaterMat.dispose();
      particleGeo.dispose();
      particleMat.dispose();
      renderer.dispose();
    };
  }, []);

  return (
    <div className="vessel-3d-wrapper">
      <div className="vessel-3d-canvas-container" ref={mountRef} />
      <div className="vessel-legend">
        <div className="legend-item">
          <span className="legend-dot inlet-dot"></span>
          <span>Inlet Pipe</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot seal-dot"></span>
          <span>Seal Ring</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot coil-dot"></span>
          <span>Heater Coil</span>
        </div>
      </div>
    </div>
  );
};
