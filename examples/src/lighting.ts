import * as Stats from 'stats.js';
import * as THREE from 'three';
import { Clock, Material, Mesh, PCFShadowMap, PerspectiveCamera, PlaneGeometry, PointLight, RepeatWrapping, SphereGeometry, TextureLoader, Vector3, WebGLRenderer, MeshStandardMaterial } from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { NodeShaderMaterial, varyingAttributes } from '../../src/index';
import { float, rgb, textureSampler2d, uniformFloat } from '../../src/lib/dsl';
import { colorToNormal } from '../../src/lib/effects/normal-mapping';
import { standardMaterial } from '../../src/lib/effects/physical';
import { triplanarMapping } from '../../src/lib/effects/triplannar-mapping';
import { sin } from '../../src/lib/functions';
import { translateY } from '../../src/lib/transformation/transforms';
import { UniformFloatNode } from '../../src/lib/uniforms';

export function init() {
  const bricksTexture = new TextureLoader().load('./assets/textures/bricks/Bricks075A_1K_Color.png')
  bricksTexture.wrapS = RepeatWrapping
  bricksTexture.wrapT = RepeatWrapping

  const bricksTextureNormal = new TextureLoader().load('./assets/textures/bricks/Bricks075A_1K_NormalGL.png')
  bricksTextureNormal.wrapS = RepeatWrapping
  bricksTextureNormal.wrapT = RepeatWrapping

  const bricksTextureRoughness = new TextureLoader().load('./assets/textures/bricks/Bricks075A_1K_Roughness.png')
  bricksTextureRoughness.wrapS = RepeatWrapping
  bricksTextureRoughness.wrapT = RepeatWrapping

  var stats = new Stats();
  stats.showPanel(0); // 0: fps, 1: ms, 2: mb, 3+: custom


  const outputContainer = document.getElementById('output');

  outputContainer.appendChild(stats.dom);


  if (outputContainer == null) {
    throw new Error("Missing output container element")
  }

  // renderer setup
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(0, 1);
  renderer.shadowMap.enabled = true
  renderer.shadowMap.type = PCFShadowMap
  outputContainer.appendChild(renderer.domElement);

  // camera setup
  const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 50);
  camera.position.set(0, 7, 7);
  camera.far = 100;
  camera.lookAt(new Vector3(0, 0, 0))
  camera.updateProjectionMatrix();

  function onResize(camera: PerspectiveCamera, renderer: WebGLRenderer) {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);

    renderer.setSize(
      window.innerWidth,
      window.innerHeight
    );
  }

  window.addEventListener('resize', () => onResize(camera, renderer), false);
  onResize(camera, renderer);

  new OrbitControls(camera, renderer.domElement);

  const clock = new Clock()

  const scene = new THREE.Scene()

  const sphere = new SphereGeometry(5, 30, 15)

  const uniformTime = uniformFloat("time")

  const sampler = textureSampler2d(bricksTexture)
  const normalSampler = textureSampler2d(bricksTextureNormal)

  const triplanarScale = float(0.15)

  const color = triplanarMapping(sampler, triplanarScale).rgb()

  const normalSample = triplanarMapping(normalSampler, triplanarScale).xyz();

  const normal = colorToNormal(normalSample, 0.7)

  let material: Material = new NodeShaderMaterial({
    //color: standardMaterial({ color, normal }),
    color: standardMaterial({
      color: sampler.sample(varyingAttributes.uv).rgb(),
      normal: colorToNormal(normalSampler.sample(varyingAttributes.uv), 0.9)
    })
    //transform: oscilate(uniformTime)
  })
  material = new MeshStandardMaterial({ map: bricksTexture, normalMap: bricksTextureNormal, roughness: 0.2, metalness: 0 })

  material.onBeforeCompile = (s) => {
    console.log(s.fragmentShader)
  }
  // If the attributes.tangent is used, tangents have to be manually computed on the geometry. 
  sphere.computeTangents()

  const mesh = new Mesh(sphere, material)
  mesh.castShadow = true
  mesh.position.set(0, 0, 0)
  scene.add(mesh)

  const pointlight = new PointLight(null, 0.2)
  pointlight.position.set(10, 10, 5)
  pointlight.castShadow = true
  scene.add(pointlight)

  const pointlight2 = new PointLight(null, 0.1)
  pointlight2.position.set(10, 10, 3)
  pointlight2.castShadow = true
  scene.add(pointlight2)

  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.4);
  directionalLight.position.x = -0.5
  directionalLight.castShadow = true
  scene.add(directionalLight);

  const spotLight = new THREE.SpotLight(0xffffff, 0.5);
  spotLight.position.set(1, 15, 0);

  spotLight.castShadow = true;

  spotLight.shadow.mapSize.width = 1024;
  spotLight.shadow.mapSize.height = 1024;

  spotLight.shadow.camera.near = 1;
  spotLight.shadow.camera.far = 4000;
  spotLight.shadow.camera.fov = 30;
  scene.add(spotLight)

  const hemilight = new THREE.HemisphereLight(0xffffbb, 0x080820, 0.3);
  scene.add(hemilight);

  scene.add(createPlane())

  function render() {
    stats.begin()
    //material.uniforms.time.value = clock.getElapsedTime()

    renderer.render(scene, camera)
    stats.end()
    requestAnimationFrame(render)

  }
  render()

}

function createPlane() {
  const plane = new PlaneGeometry(40, 40);

  let material: Material;
  material = new NodeShaderMaterial({
    color: standardMaterial({ color: rgb(0xcccccc) }),
    uniforms: {
      time: { value: 0 }
    }
  })
  //material = new MeshStandardMaterial({ color: 0xcccccc })


  const mesh = new Mesh(plane, material)
  mesh.receiveShadow = true

  mesh.translateY(-5)
  mesh.rotateX(-Math.PI / 2)
  return mesh;
}

function oscilate(uniformTime: UniformFloatNode) {
  return translateY(sin(uniformTime.multiply(float(5))))
}