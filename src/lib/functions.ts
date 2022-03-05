import { Compiler, ShaderNode } from './compiler';
import { float } from './dsl';
import { BaseType } from './nodes';
import {
  FloatNode,
  IFloatNode,
  Mat2Node,
  Mat3Node,
  Mat4Node,
  Vec2Node,
  Vec3Node,
  Vec4Node,
} from './types';

// Find math functions here
// https://www.shaderific.com/glsl-functions

export class FloatBiFuncNode extends FloatNode {
  constructor(
    private readonly a: IFloatNode,
    private readonly b: IFloatNode,
    private readonly func: string
  ) {
    super();
  }
  public compile(c: Compiler) {
    const k = c.variable();
    return {
      chunk: `
        float float_${k} = ${this.func}(${c.get(this.a)}, ${c.get(this.b)});
      `,
      out: `float_${k}`,
    };
  }
}

export class FloatPowNode extends FloatBiFuncNode {
  constructor(a: IFloatNode, b: IFloatNode) {
    super(a, b, 'pow');
  }
}

export class FloatMaxNode extends FloatBiFuncNode {
  constructor(a: IFloatNode, b: IFloatNode) {
    super(a, b, 'max');
  }
}

export class FloatMinNode extends FloatBiFuncNode {
  constructor(a: IFloatNode, b: IFloatNode) {
    super(a, b, 'min');
  }
}

export class FloatDistanceNode extends FloatBiFuncNode {
  constructor(a: IFloatNode, b: IFloatNode) {
    super(a, b, 'distance');
  }
}

export class FloatDotNode extends FloatBiFuncNode {
  constructor(a: IFloatNode, b: IFloatNode) {
    super(a, b, 'dot');
  }
}

function unaryFunction<T extends ShaderNode<T>>(name: string, value: T): T {
  return new (class extends Object.getPrototypeOf(value).constructor {
    public compile(c: Compiler) {
      return { out: `${name}(${c.get(value)})` };
    }
  })() as unknown as T;
}

// Functions with a single numeric argument

export function inverse<T extends Mat2Node | Mat3Node | Mat4Node>(x: T): T {
  return unaryFunction('inverse', x);
}

export function transpose<T extends Mat2Node | Mat3Node | Mat4Node>(x: T): T {
  return unaryFunction('transpose', x);
}

export function radians<T extends FloatNode | Vec2Node | Vec3Node | Vec4Node>(
  x: T
): T {
  return unaryFunction('radians', x);
}

export function degrees<T extends FloatNode | Vec2Node | Vec3Node | Vec4Node>(
  x: T
): T {
  return unaryFunction('degrees', x);
}

export function sin<T extends FloatNode | Vec2Node | Vec3Node | Vec4Node>(
  x: T
): T {
  return unaryFunction('sin', x);
}

export function cos<T extends FloatNode | Vec2Node | Vec3Node | Vec4Node>(
  x: T
): T {
  return unaryFunction('cos', x);
}

export function tan<T extends FloatNode | Vec2Node | Vec3Node | Vec4Node>(
  x: T
): T {
  return unaryFunction('tan', x);
}

export function asin<T extends FloatNode | Vec2Node | Vec3Node | Vec4Node>(
  x: T
): T {
  return unaryFunction('asin', x);
}

export function acos<T extends FloatNode | Vec2Node | Vec3Node | Vec4Node>(
  x: T
): T {
  return unaryFunction('acos', x);
}

export function atan<T extends FloatNode | Vec2Node | Vec3Node | Vec4Node>(
  yOverX: T
): T {
  return unaryFunction('atan', yOverX);
}

export function length<T extends FloatNode | Vec2Node | Vec3Node | Vec4Node>(
  x: T
): FloatNode {
  return anyArgsFunction(FloatNode, 'length', x);
}

export function normalize<T extends FloatNode | Vec2Node | Vec3Node | Vec4Node>(
  x: T
): T {
  return unaryFunction('normalize', x);
}

export function exp<T extends FloatNode | Vec2Node | Vec3Node | Vec4Node>(
  x: T
): T {
  return unaryFunction('exp', x);
}

export function log<T extends FloatNode | Vec2Node | Vec3Node | Vec4Node>(
  x: T
): T {
  return unaryFunction('log', x);
}

export function exp2<T extends FloatNode | Vec2Node | Vec3Node | Vec4Node>(
  x: T
): T {
  return unaryFunction('exp2', x);
}

export function log2<T extends FloatNode | Vec2Node | Vec3Node | Vec4Node>(
  x: T
): T {
  return unaryFunction('log2', x);
}

export function sqrt<T extends FloatNode | Vec2Node | Vec3Node | Vec4Node>(
  x: T
): T {
  return unaryFunction('sqrt', x);
}

export function inversesqrt<
  T extends FloatNode | Vec2Node | Vec3Node | Vec4Node
>(x: T): T {
  return unaryFunction('inversesqrt', x);
}

export function abs<T extends FloatNode | Vec2Node | Vec3Node | Vec4Node>(
  x: T
): T {
  return unaryFunction('abs', x);
}

export function dFdx<T extends FloatNode | Vec2Node | Vec3Node | Vec4Node>(
  x: T
): T {
  return unaryFunction('dFdx', x);
}

export function dFdy<T extends FloatNode | Vec2Node | Vec3Node | Vec4Node>(
  x: T
): T {
  return unaryFunction('dFdy', x);
}

export function dFdxCoarse<T extends FloatNode | Vec2Node | Vec3Node | Vec4Node>(
  x: T
): T {
  return unaryFunction('dFdxCoarse', x);
}

export function dFdyCoarse<T extends FloatNode | Vec2Node | Vec3Node | Vec4Node>(
  x: T
): T {
  return unaryFunction('dFdyCoarse', x);
}

export function dFdxFine<T extends FloatNode | Vec2Node | Vec3Node | Vec4Node>(
  x: T
): T {
  return unaryFunction('dFdxFine', x);
}

export function dFdyFine<T extends FloatNode | Vec2Node | Vec3Node | Vec4Node>(
  x: T
): T {
  return unaryFunction('dFdyFine', x);
}

export function sign<T extends FloatNode | Vec2Node | Vec3Node | Vec4Node>(
  x: T
): T {
  return unaryFunction('sign', x);
}

export function floor<T extends FloatNode | Vec2Node | Vec3Node | Vec4Node>(
  x: T
): T {
  return unaryFunction('floor', x);
}

export function fract<T extends FloatNode | Vec2Node | Vec3Node | Vec4Node>(
  x: T
): T {
  return unaryFunction('fract', x);
}

export function ceil<T extends FloatNode | Vec2Node | Vec3Node | Vec4Node>(
  x: T
): T {
  return unaryFunction('ceil', x);
}

// Functions with more than one argument

function anyArgsFunction<R extends ShaderNode<R>>(
  returnType: BaseType<R>,
  name: string,
  ...args: readonly ShaderNode<string>[]
): R {
  return new (class extends returnType {
    // @ts-expect-error
    public compile(c: Compiler) {
      const argsList = args.map((a) => c.get(a)).join(', ');
      return { out: `${name}(${argsList})` };
    }
  })() as unknown as R;
}

export function clamp<
  T extends FloatNode | Vec2Node | Vec3Node | Vec4Node,
  U extends T | FloatNode
>(x: T, minVal: U, maxVal: U): T {
  return anyArgsFunction(
    Object.getPrototypeOf(x).constructor,
    'clamp',
    x,
    minVal,
    maxVal
  );
}

export function mix<
  T extends FloatNode | Vec2Node | Vec3Node | Vec4Node,
  U extends T | FloatNode
>(x: T, y: T, a: U): T {
  return anyArgsFunction(Object.getPrototypeOf(x).constructor, 'mix', x, y, a);
}

export function step<
  T extends FloatNode | Vec2Node | Vec3Node | Vec4Node,
  U extends T | FloatNode
>(edge: U, x: T): T {
  return anyArgsFunction(Object.getPrototypeOf(x).constructor, 'step', edge, x);
}

export function smoothstep<
  T extends FloatNode | Vec2Node | Vec3Node | Vec4Node,
  U extends T | FloatNode
>(edge0: U, edge1: U, x: T): T {
  return anyArgsFunction(
    Object.getPrototypeOf(x).constructor,
    'smoothstep',
    edge0,
    edge1,
    x
  );
}

export function pow<T extends FloatNode | Vec2Node | Vec3Node | Vec4Node>(
  x: T,
  y: T
): T {
  return anyArgsFunction(Object.getPrototypeOf(x).constructor, 'pow', x, y);
}

export function max<T extends FloatNode | Vec2Node | Vec3Node | Vec4Node>(
  x: T,
  y: T
): T {
  return anyArgsFunction(Object.getPrototypeOf(x).constructor, 'max', x, y);
}

export function min<T extends FloatNode | Vec2Node | Vec3Node | Vec4Node>(
  x: T,
  y: T
): FloatNode {
  return anyArgsFunction(Object.getPrototypeOf(x).constructor, 'min', x, y);
}

export function dot<T extends FloatNode | Vec2Node | Vec3Node | Vec4Node>(
  x: T,
  y: T
): FloatNode {
  return anyArgsFunction(FloatNode, 'dot', x, y);
}

export function cross<T extends FloatNode | Vec2Node | Vec3Node | Vec4Node>(
  x: T,
  y: T
): T {
  return anyArgsFunction(Object.getPrototypeOf(x).constructor, 'cross', x, y);
}

export function saturate<T extends FloatNode | Vec2Node | Vec3Node | Vec4Node>(
  a: T
): T {
  return clamp(a, float(0), float(1));
}

