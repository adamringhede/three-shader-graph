import { uniforms } from '../common';
import {
  float,
  int,
  negVec3,
  rgb,
  rgba,
  varyingArray,
  vec4,
} from '../dsl';
import { abs, dot, exp2, FloatMaxNode, FloatMinNode, max, normalize, pow, saturate, sqrt, dFdx, dFdy, min } from '../functions';
import {
  uniformDirectionalLights,
  uniformDirectionalLightShadows,
  uniformDirectionalShadowMap,
  uniformDirectionalShadowMatrix,
  uniformHemisphereLights,
  uniformPointLights,
  uniformPointLightShadows,
  uniformPointShadowMap,
  uniformPointShadowMatrix,
  uniformSpotLights,
  uniformSpotLightShadows,
  uniformSpotShadowMap,
  uniformSpotShadowMatrix,
} from '../lights';
import { selectPreCompile } from '../nodes';
import { transformed, varyingTransformed } from '../transformed';
import { FloatNode, RgbNode, Vec3Node } from '../types';

import {
  BRDF_Lambert,
  Geometry,
  getDirectionalLightInfo,
  getHemisphereLightIrradiance,
  getPointLightInfo,
  getSpotLightInfo,
  RECIPROCAL_PI,
} from './common-material';
import { GetPointShadowNode } from './point-shadow';
import { GetShadowNode } from './shadow';
import { IncidentLight } from './common-material';
import { DirectionalLight } from '../lights';

type PhysicalMaterial = {
  readonly diffuseColor: RgbNode;
  readonly specularColor: RgbNode;
  readonly roughness: FloatNode;
};

const worldPosition = uniforms.modelMatrix.multiplyVec(transformed.position);
const shadowWorldNormal = normalize(
  vec4(transformed.normal, 0.0).multiplyMat(uniforms.viewMatrix).xyz()
);

function getDirectSpecular(irradiance: Vec3Node, directLight: IncidentLight, geometry: Geometry, material: PhysicalMaterial): Vec3Node {
  const specularF90 = float(1)
  return irradiance.multiply(BRDF_GGX(directLight.direction, geometry.viewDir, geometry.normal, material.specularColor, specularF90, float(material.roughness)))
}

function calculatePointLight(
  geometry: Geometry,
  material: PhysicalMaterial
): Vec3Node {
  const pointShadowCoords = uniformPointLightShadows.map((p, i) => {
    const shadowWorldPosition = worldPosition.add(
      vec4(shadowWorldNormal.multiplyScalar(p.shadowNormalBias), 0)
    );
    return uniformPointShadowMatrix.get(i).multiplyVec(shadowWorldPosition);
  });
  const vPointShadowCoord = varyingArray(pointShadowCoords);

  const directDiffuse = uniformPointLights.sum((light, i) => {
    const pointLightShadow = uniformPointLightShadows.get(i);

    const getShadowNode = new GetPointShadowNode(
      uniformPointShadowMap.get(i),
      pointLightShadow.shadowMapSize,
      pointLightShadow.shadowBias,
      pointLightShadow.shadowRadius,
      vPointShadowCoord.get(i),
      pointLightShadow.shadowCameraNear,
      pointLightShadow.shadowCameraFar
    );
    const shadowFactor = selectPreCompile(
      int(uniformPointShadowMap.limit).gt(i),
      getShadowNode,
      float(1.0))

    const directLight = getPointLightInfo(light, geometry);
    const dotNL = saturate(dot(geometry.normal, directLight.direction));
    const irradiance = dotNL.multiplyVec3(light.color);

    const directSpecular = getDirectSpecular(irradiance, directLight, geometry, material)

    return irradiance
      .multiply(BRDF_Lambert(material.diffuseColor))
      .multiplyScalar(shadowFactor)
      .add(directSpecular)
  });
  return directDiffuse;
}

function calculateSpotLight(
  geometry: Geometry,
  material: PhysicalMaterial
): Vec3Node {
  const spotShadowCoords = uniformSpotLightShadows.map((p, i) => {
    const shadowWorldPosition = worldPosition.add(
      vec4(shadowWorldNormal.multiplyScalar(p.shadowNormalBias), 0)
    );
    return uniformSpotShadowMatrix.get(i).multiplyVec(shadowWorldPosition);
  });
  const vSpotShadowCoord = varyingArray(spotShadowCoords);

  const directDiffuse = uniformSpotLights.sum((light, i) => {
    const spotLightShadow = uniformSpotLightShadows.get(i);

    const getShadowNode = new GetShadowNode(
      uniformSpotShadowMap.get(i),
      spotLightShadow.shadowMapSize,
      spotLightShadow.shadowBias,
      spotLightShadow.shadowRadius,
      vSpotShadowCoord.get(i)
    )
    const shadowFactor = selectPreCompile(
      int(uniformSpotShadowMap.limit).gt(i),
      getShadowNode,
      float(1.0))

    const directLight = getSpotLightInfo(light, geometry);
    const dotNL = saturate(dot(geometry.normal, directLight.direction));
    const irradiance = dotNL.multiplyVec3(light.color);

    const directSpecular = getDirectSpecular(irradiance, directLight, geometry, material)

    return irradiance
      .multiply(BRDF_Lambert(material.diffuseColor))
      .multiplyScalar(shadowFactor)
    //.add(directSpecular);
  });
  return directDiffuse;
}

function calculateDirectionalLight(
  geometry: Geometry,
  material: PhysicalMaterial
): Vec3Node {
  const directionalShadowCoords = uniformDirectionalLightShadows.map((p, i) => {
    const shadowWorldPosition = worldPosition.add(
      vec4(shadowWorldNormal.multiplyScalar(p.shadowNormalBias), 0)
    );
    return uniformDirectionalShadowMatrix
      .get(i)
      .multiplyVec(shadowWorldPosition);
  });
  const vDirectionalShadowCoord = varyingArray(directionalShadowCoords);

  const directDiffuse = uniformDirectionalLights.sum((light, i) => {
    const directionalLightShadow = uniformDirectionalLightShadows.get(i);

    const getShadowNode = new GetShadowNode(
      uniformDirectionalShadowMap.get(i),
      directionalLightShadow.shadowMapSize,
      directionalLightShadow.shadowBias,
      directionalLightShadow.shadowRadius,
      vDirectionalShadowCoord.get(i)
    )
    const shadowFactor = selectPreCompile(
      int(uniformDirectionalShadowMap.limit).gt(i),
      getShadowNode,
      float(1.0))

    const directLight = getDirectionalLightInfo(light, geometry);
    const dotNL = saturate(dot(geometry.normal, directLight.direction));
    const irradiance = dotNL.multiplyVec3(light.color);

    const directSpecular = getDirectSpecular(irradiance, directLight, geometry, material)

    return irradiance
      .multiply(BRDF_Lambert(material.diffuseColor))
      .multiplyScalar(shadowFactor)
      .add(directSpecular);
  });
  return directDiffuse;
}

function pow2(v: FloatNode): FloatNode {
  return v.multiply(v)
}

function F_Schlick(f0: Vec3Node, f90: FloatNode, dotVH: FloatNode) {
  const fresnel = exp2(float(-5.55473).multiply(dotVH).subtract(float(6.98316)).multiply(dotVH))
  return f0.multiplyScalar(float(1).subtract(fresnel)).addScalar(f90.multiply(fresnel))
}

const EPSILON = float(0.000001)

function V_GGX_SmithCorrelated(alpha: FloatNode, dotNL: FloatNode, dotNV: FloatNode): FloatNode {
  const a2 = pow2(alpha)
  const a2inv = float(1).subtract(a2)
  const gv = dotNL.multiply(sqrt(a2inv.multiply(pow2(dotNV)).add(a2)))
  const gl = dotNV.multiply(sqrt(a2inv.multiply(pow2(dotNL)).add(a2)))
  return float(0.5).divide(max(gv.add(gl), EPSILON))
}

function D_GGX(alpha: FloatNode, dotNH: FloatNode): FloatNode {
  const a2 = pow2(alpha)
  const denom = pow2(dotNH).multiply(a2.subtract(float(1))).add(float(1))
  return RECIPROCAL_PI.multiply(a2).divide(pow2(denom))
}

function BRDF_GGX(lightDir: Vec3Node, viewDir: Vec3Node, normal: Vec3Node, f0: Vec3Node, f90: FloatNode, roughness: FloatNode): Vec3Node {
  const alpha = pow2(roughness)
  const halfDir = normalize(lightDir.add(viewDir))

  const dotNL = saturate(dot(normal, lightDir))
  const dotNV = saturate(dot(normal, viewDir));
  const dotNH = saturate(dot(normal, halfDir));
  const dotVH = saturate(dot(viewDir, halfDir));

  const F = F_Schlick(f0, f90, dotVH)
  const V = V_GGX_SmithCorrelated(alpha, dotNL, dotNV)
  const D = D_GGX(alpha, dotNH)
  return F.multiplyScalar(V.multiply(D))
}

function calculateHemisphereLight(
  geometry: Geometry,
  material: PhysicalMaterial
) {
  return uniformHemisphereLights
    .sum((light) => getHemisphereLightIrradiance(light, geometry.normal))
    .multiply(BRDF_Lambert(material.diffuseColor));
}

export type StandardMaterialParameters = {
  readonly color: RgbNode;
  readonly emissive: RgbNode;
  readonly emissiveIntensity: FloatNode;
  readonly normal: Vec3Node | null;
  readonly roughness: FloatNode
};

const standardMaterialParametersDefaults: StandardMaterialParameters = {
  color: rgb(0x000000),
  emissive: rgb(0x000000),
  emissiveIntensity: float(1),
  normal: varyingTransformed.normal,
  roughness: float(1)
};

function getRoughness(geometry: Geometry, roughnessFactor: FloatNode): FloatNode {
  const dxy = max(abs(dFdx(geometry.normal)), abs(dFdy(geometry.normal)))
  const geometryRoughness = max(max(dxy.x(), dxy.y()), dxy.z())
  return min(max(roughnessFactor, float(0.0525)).add(geometryRoughness), float(1.0))
}

export function standardMaterial(params: Partial<StandardMaterialParameters>) {
  const { color, emissive, emissiveIntensity, normal, roughness } = {
    ...standardMaterialParametersDefaults,
    ...params,
  };

  const vPos = varyingTransformed.mvPosition.xyz()

  const geometry = {
    position: vPos,
    normal: normal,
    viewDir: normalize(negVec3(vPos)),
  } as Geometry;

  const material = {
    diffuseColor: color,
    specularColor: rgb(0xffffff),
    roughness: getRoughness(geometry, float(0.2))
  } as PhysicalMaterial;

  const directDiffuse = [
    calculatePointLight(geometry, material),
    calculateDirectionalLight(geometry, material),
    calculateSpotLight(geometry, material),
  ].reduce((a, v) => a.add(v));

  const indirectDiffuse = calculateHemisphereLight(geometry, material);

  const totalDiffuse = directDiffuse.add(indirectDiffuse);

  const outgoingLight =
    emissive != null
      ? totalDiffuse.add(emissive.multiplyScalar(emissiveIntensity))
      : totalDiffuse;

  return rgba(outgoingLight, 1);
}
