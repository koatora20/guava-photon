---
name: guava-photon
description: "Node.js互換AOTコンパイラ × WASM SIMD 128-bit テンソルエンジン。session-load v8の100万次元認知状態ベクトルを2x高速で処理。"
---

# guava-photon — Session-Load Tensor Engine

> 物理法則が許す速度の10兆分の1も出ていない → 2x/10x/1000xへの道

## Overview

guava-photonは、session-load v8の100万次元認知パイプラインをWASM SIMD 128-bitで高速化するAOTコンパイラ+テンソルエンジン。

## Architecture

```
Layer 1: AOT Compilation
  TypeScript subset → WAT → WASM binary
  compiler.mjs: Lexer → Parser → Codegen (for-loops, assignments, functions)

Layer 2: SIMD Tensor Engine
  Hand-written simd-tensor.wat
  f64x2.mul / f64x2.add / v128.load — 2 values per cycle
  
Layer 3: Session-Load Cognitive Kernel
  8 functions: dotProduct, magnitude, cosineSimilarity, expApprox,
  qDecay, weightUpdate, parityTensor, viralCoefficient
```

## Proven Performance

| Dimension | SIMD vs Node.js | SIMD vs Scalar |
|---|---|---|
| 1K | 1.46x | 1.24x |
| 10K | 1.87x | 1.91x |
| 50K | 1.91x | 2.03x |
| 500K | 2.00x | 2.16x |
| 1M | 1.87x | 2.00x |

## Usage

### Step 1: Compile TypeScript → WASM
```bash
node src/compiler.mjs src/tensor.ts        # TS → WAT
wat2wasm src/tensor.wat -o tensor.wasm      # WAT → WASM
```

### Step 2: Load and Execute
```javascript
import { readFileSync } from 'fs';
const buf = readFileSync('tensor.wasm');
const { instance } = await WebAssembly.instantiate(buf);

// Use cognitive functions
const result = instance.exports.parityTensor(0.8, 0.3, 0.6, 0.1);
const k = instance.exports.viralCoefficient(2.0, 0.65);
```

### Step 3: SIMD Batched Tensors
```javascript
const simdBuf = readFileSync('/tmp/simd-tensor.wasm');
const { instance: simd } = await WebAssembly.instantiate(simdBuf);

// Fill memory with 1M-dim vectors
const mem = new Float64Array(simd.exports.memory.buffer);
// ... fill vectors A and B ...

// SIMD dot product at 2x Node.js speed
const dot = simd.exports.simdDotProduct(0, 8_000_000, 1_000_000);
```

## GAN-TDD Test Suite

```bash
# All tests
node --test test/tensor.test.mjs benchmarks/simd-dotprod.mjs

# Results: 20/20 PASS (6 suites)
```

## Security

- WAT: no data sections (zero injection surface)
- Binary: 333 bytes + 810 bytes (minimal attack surface)
- Memory: bounded 256 pages = 16MB (DoS resistant)
- guard-scanner: PASS

## References

- Bremermann (1962): Physical computation limits
- Margolus-Levitin (1998): Maximum speed of dynamical evolution
- Landauer (1961): Irreversibility and heat generation
- WASM SIMD Spec: github.com/WebAssembly/simd
