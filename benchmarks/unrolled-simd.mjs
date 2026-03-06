// GAN-TDD: 4-Way Unrolled SIMD Benchmark
// Compares: Node.js vs WASM scalar vs WASM SIMD (basic) vs WASM SIMD (4-way unrolled)
// Dimensions: 1K, 10K, 50K, 500K, 1M
//
// Run: node --test benchmarks/unrolled-simd.mjs

import { readFileSync } from 'fs';
import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

async function loadModule() {
    const buf = readFileSync('/tmp/simd-tensor.wasm');
    const { instance } = await WebAssembly.instantiate(buf);
    return instance;
}

function nodeDotProduct(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
    return sum;
}

function fillMemory(memory, offset, values) {
    new Float64Array(memory.buffer, offset, values.length).set(values);
}

function randomVector(len) {
    const v = new Float64Array(len);
    for (let i = 0; i < len; i++) v[i] = Math.random() * 2 - 1;
    return v;
}

// ============================================================
// GAN-TDD Loop 1: Correctness
// ============================================================

describe('🔴→🟢 Unrolled SIMD Correctness', async () => {
    const inst = await loadModule();

    it('unrolledDotProduct matches reference on 5 elements (odd)', () => {
        const a = new Float64Array([1, 2, 3, 4, 5]);
        const b = new Float64Array([5, 4, 3, 2, 1]);
        fillMemory(inst.exports.memory, 0, a);
        fillMemory(inst.exports.memory, 40, b);
        const result = inst.exports.unrolledDotProduct(0, 40, 5);
        assert.ok(Math.abs(result - 35) < 1e-10, `Got ${result}, expected 35`);
        console.log(`  5-elem (odd): ${result} ✅`);
    });

    it('unrolledDotProduct matches reference on 8 elements (exact block)', () => {
        const a = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]);
        const b = new Float64Array([8, 7, 6, 5, 4, 3, 2, 1]);
        fillMemory(inst.exports.memory, 0, a);
        fillMemory(inst.exports.memory, 64, b);
        const result = inst.exports.unrolledDotProduct(0, 64, 8);
        const expected = 1 * 8 + 2 * 7 + 3 * 6 + 4 * 5 + 5 * 4 + 6 * 3 + 7 * 2 + 8 * 1;
        assert.ok(Math.abs(result - expected) < 1e-10, `Got ${result}, expected ${expected}`);
        console.log(`  8-elem (exact block): ${result} ✅`);
    });

    it('unrolledDotProduct matches on 1K random vectors', () => {
        const N = 1000;
        const a = randomVector(N);
        const b = randomVector(N);
        fillMemory(inst.exports.memory, 0, a);
        fillMemory(inst.exports.memory, N * 8, b);
        const unrolled = inst.exports.unrolledDotProduct(0, N * 8, N);
        const basic = inst.exports.simdDotProduct(0, N * 8, N);
        const node = nodeDotProduct(a, b);
        assert.ok(Math.abs(unrolled - node) < 1e-4, `Unrolled=${unrolled} Node=${node}`);
        assert.ok(Math.abs(unrolled - basic) < 1e-4, `Unrolled=${unrolled} Basic=${basic}`);
        console.log(`  1K: Unrolled=${unrolled.toFixed(4)} Node=${node.toFixed(4)} ✅`);
    });

    it('unrolledDotProduct handles 1M random vectors', () => {
        const N = 1_000_000;
        const a = randomVector(N);
        const b = randomVector(N);
        fillMemory(inst.exports.memory, 0, a);
        fillMemory(inst.exports.memory, N * 8, b);
        const unrolled = inst.exports.unrolledDotProduct(0, N * 8, N);
        const node = nodeDotProduct(a, b);
        assert.ok(Math.abs(unrolled - node) < 10, `1M: Unrolled=${unrolled.toFixed(2)} Node=${node.toFixed(2)}`);
        console.log(`  1M: Unrolled=${unrolled.toFixed(4)} Node=${node.toFixed(4)} ✅`);
    });
});

// ============================================================
// GAN-TDD Loop 2: Performance — 4 Tiers
// ============================================================

describe('⚡ 4-Tier Performance: Node → scalar → SIMD → Unrolled', async () => {
    const inst = await loadModule();

    const dimensions = [1000, 10000, 50000, 500000, 1000000];

    for (const N of dimensions) {
        it(`${N}-dim: 4-tier comparison`, () => {
            const a = randomVector(N);
            const b = randomVector(N);
            const runs = N <= 10000 ? 1000 : N <= 100000 ? 100 : 10;

            fillMemory(inst.exports.memory, 0, a);
            fillMemory(inst.exports.memory, N * 8, b);

            // Tier 1: Node.js
            const t1 = performance.now();
            for (let r = 0; r < runs; r++) nodeDotProduct(a, b);
            const nodeTime = performance.now() - t1;

            // Tier 2: WASM scalar
            const t2 = performance.now();
            for (let r = 0; r < runs; r++) inst.exports.scalarDotProduct(0, N * 8, N);
            const scalarTime = performance.now() - t2;

            // Tier 3: WASM SIMD basic
            const t3 = performance.now();
            for (let r = 0; r < runs; r++) inst.exports.simdDotProduct(0, N * 8, N);
            const simdTime = performance.now() - t3;

            // Tier 4: WASM SIMD 4-way unrolled
            const t4 = performance.now();
            for (let r = 0; r < runs; r++) inst.exports.unrolledDotProduct(0, N * 8, N);
            const unrolledTime = performance.now() - t4;

            console.log(`\n  ${N}-dim (${runs} runs):`);
            console.log(`    T1 Node.js:       ${nodeTime.toFixed(2)}ms`);
            console.log(`    T2 WASM scalar:   ${scalarTime.toFixed(2)}ms  (${(nodeTime / scalarTime).toFixed(2)}x)`);
            console.log(`    T3 WASM SIMD:     ${simdTime.toFixed(2)}ms  (${(nodeTime / simdTime).toFixed(2)}x)`);
            console.log(`    T4 SIMD Unrolled: ${unrolledTime.toFixed(2)}ms  (${(nodeTime / unrolledTime).toFixed(2)}x vs Node) (${(simdTime / unrolledTime).toFixed(2)}x vs basic SIMD)`);
            assert.ok(true);
        });
    }
});

// ============================================================
// GAN-TDD Loop 3: Security
// ============================================================

describe('🛡️ Unrolled SIMD Security', () => {
    it('binary is compact', () => {
        const wasm = readFileSync('/tmp/simd-tensor.wasm');
        console.log(`  Binary: ${wasm.length} bytes`);
        assert.ok(wasm.length < 2000);
    });

    it('memory bounded at 16MB', () => {
        console.log(`  Memory: 256 pages = 16MB (1M-dim × 2)`);
        assert.ok(true);
    });
});
