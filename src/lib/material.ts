import { IUniform, ShaderMaterial } from 'three';
import * as THREE from 'three';

import { ShaderGraph } from './compiler';
import { rgba } from './dsl';
import { ConstantMat4Node, IRgbaNode, Mat4Node } from './types';

const identityMat4 = new ConstantMat4Node(
  1,
  0,
  0,
  0,
  0,
  1,
  0,
  0,
  0,
  0,
  1,
  0,
  0,
  0,
  0,
  1
);

type NodeShaderMaterialParameters = {
  readonly color?: IRgbaNode;
  readonly transform?: Mat4Node;
  readonly uniforms?: { readonly [uniform: string]: IUniform };
  readonly transparent?: boolean;
};

const nodeShaderMaterialDefaults = {
  color: rgba(0x000),
  transform: identityMat4,
  uniforms: {},
  transparent: false,
} as Required<NodeShaderMaterialParameters>;

export class NodeShaderMaterial extends ShaderMaterial {
  constructor(params: NodeShaderMaterialParameters) {
    const paramsWithDefaults = {
      ...nodeShaderMaterialDefaults,
      ...params,
    };

    const shaders = new ShaderGraph({
      color: paramsWithDefaults.color,
      transform: paramsWithDefaults.transform,
    }).compile();

    const uniforms = {
      ...shaders.uniforms,
      ...paramsWithDefaults.uniforms,
      ...THREE.UniformsLib.fog,
      ...THREE.UniformsLib.lights,
    }

    super({
      fragmentShader: shaders.fragmentShader,
      vertexShader: shaders.vertexShader,
      transparent: paramsWithDefaults.transparent,
      lights: true,
      fog: true, // Necessary for fog uniforms to be set based on fog set up on the scene.
      uniforms
    });


  }
}
