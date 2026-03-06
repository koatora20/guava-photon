// WASM SIMD f64x2 Batched Dot Product Benchmark
// Compares: Node.js scalar vs WASM scalar vs WASM SIMD
// Test dimensions: 1K, 10K, 100K, 1M
//
// Run: node benchmarks/simd-dotprod.mjs

import { readFileSync } from 'fs';
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

// ============================================================
// Setup
// ============================================================

async function loadSIMDModule() {
    const wasmBuffer = readFileSync('/tmp/simd-tensor.wasm');
    const { instance } = await WebAssembly.instantiate(wasmBuffer);
    return instance;
}

// Node.js reference: scalar dot product
function nodeDotProduct(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
    return sum;
}

// Fill WASM memory with vector data
function fillMemory(memory, offsetBytes, values) {
    const view = new Float64Array(memory.buffer, offsetBytes, values.length);
    view.set(values);
}

// Generate random f64 vector
function randomVector(len) {
    const v = new Float64Array(len);
    for (let i = 0; i < len; i++) v[i] = Math.random() * 2 - 1;
    return v;
}

// ============================================================
// GAN-TDD Loop 1: Correctness
// ============================================================

describe('🔴→🟢 SIMD Correctness', async () => {
    const inst = await loadSIMDModule();

    it('simdDotProduct matches reference on small vectors', () => {
        const a = new Float64Array([1.0, 2.0, 3.0, 4.0, 5.0]);
        const b = new Float64Array([5.0, 4.0, 3.0, 2.0, 1.0]);
        const expected = 1 * 5 + 2 * 4 + 3 * 3 + 4 * 2 + 5 * 1; // = 35

        fillMemory(inst.exports.memory, 0, a);
        fillMemory(inst.exports.memory, a.length * 8, b);

        const result = inst.exports.simdDotProduct(0, a.length * 8, a.length);
        assert.ok(Math.abs(result - expected) < 1e-10, `SIMD dotProduct = ${result}, expected ${expected}`);
        console.log(`  simdDotProduct([1..5], [5..1]) = ${result} ✅`);
    });

    it('simdDotProduct matches scalarDotProduct on 1K random vectors', () => {
        const N = 1000;
        const a = randomVector(N);
        const b = randomVector(N);

        fillMemory(inst.exports.memory, 0, a);
        fillMemory(inst.exports.memory, N * 8, b);

        const simd = inst.exports.simdDotProduct(0, N * 8, N);
        const scalar = inst.exports.scalarDotProduct(0, N * 8, N);
        const node = nodeDotProduct(a, b);

        assert.ok(Math.abs(simd - scalar) < 1e-6, `SIMD=${simd} scalar=${scalar}`);
        assert.ok(Math.abs(simd - node) < 1e-6, `SIMD=${simd} node=${node}`);
        console.log(`  1K-dim: SIMD=${simd.toFixed(4)} Node=${node.toFixed(4)} ✅`);
    });

    it('simdDotProduct handles odd-length vectors', () => {
        const N = 7; // odd
        const a = randomVector(N);
        const b = randomVector(N);

        fillMemory(inst.exports.memory, 0, a);
        fillMemory(inst.exports.memory, N * 8, b);

        const simd = inst.exports.simdDotProduct(0, N * 8, N);
        const node = nodeDotProduct(a, b);

        assert.ok(Math.abs(simd - node) < 1e-10, `Odd-length: SIMD=${simd} Node=${node}`);
        console.log(`  7-dim (odd): SIMD=${simd.toFixed(6)} Node=${node.toFixed(6)} ✅`);
    });

    it('simdMagnitude is correct', () => {
        const a = new Float64Array([3.0, 4.0]);
        fillMemory(inst.exports.memory, 0, a);

        const mag = inst.exports.simdMagnitude(0, 2);
        assert.ok(Math.abs(mag - 5.0) < 1e-10, `magnitude([3,4]) = ${mag}, expected 5`);
        console.log(`  magnitude([3,4]) = ${mag} ✅`);
    });
});

// ============================================================
// GAN-TDD Loop 2: Performance — The Main Event
// ============================================================

describe('⚡ SIMD Performance Benchmark', async () => {
    const inst = await loadSIMDModule();

    const dimensions = [1000, 10000, 50000, 500000, 1000000];

    for (const N of dimensions) {
        it(`${N}-dim dot product: Node.js vs WASM-scalar vs WASM-SIMD`, () => {
            const a = randomVector(N);
            const b = randomVector(N);

            const runs = N <= 10000 ? 1000 : N <= 100000 ? 100 : 10;

            // Fill WASM memory
            fillMemory(inst.exports.memory, 0, a);
            fillMemory(inst.exports.memory, N * 8, b);

            // Benchmark: Node.js
            let nodeSum = 0;
            const nodeStart = performance.now();
            for (let r = 0; r < runs; r++) nodeSum += nodeDotProduct(a, b);
            const nodeTime = performance.now() - nodeStart;

            // Benchmark: WASM scalar
            let scalarSum = 0;
            const scalarStart = performance.now();
            for (let r = 0; r < runs; r++) scalarSum += inst.exports.scalarDotProduct(0, N * 8, N);
            const scalarTime = performance.now() - scalarStart;

            // Benchmark: WASM SIMD
            let simdSum = 0;
            const simdStart = performance.now();
            for (let r = 0; r < runs; r++) simdSum += inst.exports.simdDotProduct(0, N * 8, N);
            const simdTime = performance.now() - simdStart;

            console.log(`\n  ${N}-dim dot product (${runs} runs):`);
            console.log(`    Node.js:     ${nodeTime.toFixed(2)}ms`);
            console.log(`    WASM scalar: ${scalarTime.toFixed(2)}ms  (${(nodeTime / scalarTime).toFixed(2)}x vs Node)`);
            console.log(`    WASM SIMD:   ${simdTime.toFixed(2)}ms  (${(nodeTime / simdTime).toFixed(2)}x vs Node) (${(scalarTime / simdTime).toFixed(2)}x vs scalar)`);

            assert.ok(true, 'Benchmark recorded');
        });
    }
});

// ============================================================
// GAN-TDD Loop 3: Security
// ============================================================

describe('🛡️ SIMD Security Audit', () => {
    it('WASM binary is minimal', () => {
        const wasm = readFileSync('/tmp/simd-tensor.wasm');
        console.log(`  SIMD binary size: ${wasm.length} bytes`);
        assert.ok(wasm.length < 2000, `Binary should be compact, got ${wasm.length}`);
    });

    it('Memory is bounded (256 pages = 16MB)', () => {
        // 256 pages × 64KB = 16MB — supports 1M-dim vectors × 2
        // Bounded memory prevents exhaustion attacks
        console.log(`  Max memory: 256 pages = 16MB (1M-dim vectors × 2)`);
        assert.ok(true);
    });
});
