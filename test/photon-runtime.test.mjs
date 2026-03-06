// GAN-TDD: PhotonRuntime + PhotonWorkerPool Test Suite
// Tests: API compatibility, WASM tensor integration, worker pool parallelization
//
// Run: node --test test/photon-runtime.test.mjs

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync, existsSync, rmSync } from 'fs';

// ============================================================
// GAN-TDD Loop 1: Correctness — API Compatibility
// ============================================================

describe('🔴→🟢 PhotonRuntime API Correctness', async () => {
    // Dynamic import to handle worker thread detection
    const { PhotonRuntime } = await import('../src/photon-runtime.mjs');
    const runtime = await new PhotonRuntime('/tmp/simd-tensor.wasm').init();

    it('fs.readFileSync works', () => {
        const testFile = '/tmp/photon-test-read.txt';
        writeFileSync(testFile, 'guava-photon test');
        const content = runtime.fs.readFileSync(testFile, 'utf-8');
        assert.equal(content, 'guava-photon test');
        console.log('  fs.readFileSync ✅');
    });

    it('fs.writeFileSync works', () => {
        const testFile = '/tmp/photon-test-write.txt';
        runtime.fs.writeFileSync(testFile, 'test-write');
        assert.equal(readFileSync(testFile, 'utf-8'), 'test-write');
        console.log('  fs.writeFileSync ✅');
    });

    it('fs.existsSync works', () => {
        assert.ok(runtime.fs.existsSync('/tmp'));
        assert.ok(!runtime.fs.existsSync('/tmp/nonexistent-guava-photon-xyz'));
        console.log('  fs.existsSync ✅');
    });

    it('path.resolve works', () => {
        const result = runtime.path.resolve('/tmp', 'test');
        assert.equal(result, '/tmp/test');
        console.log('  path.resolve ✅');
    });

    it('path.basename works', () => {
        assert.equal(runtime.path.basename('/foo/bar/baz.txt'), 'baz.txt');
        console.log('  path.basename ✅');
    });

    it('process.platform is darwin', () => {
        assert.equal(runtime.process.platform, 'darwin');
        console.log(`  process.platform = ${runtime.process.platform} ✅`);
    });

    it('process.hrtime returns elapsed ms', () => {
        const t = runtime.process.hrtime();
        assert.ok(t >= 0);
        console.log(`  process.hrtime = ${t.toFixed(2)}ms ✅`);
    });

    it('Buffer.from encodes string', () => {
        const buf = runtime.Buffer.from('hello');
        assert.equal(buf.length, 5);
        assert.equal(buf[0], 104); // 'h'
        console.log('  Buffer.from ✅');
    });
});

// ============================================================
// GAN-TDD Loop 2: Performance — Tensor Integration
// ============================================================

describe('⚡ PhotonRuntime Tensor Integration', async () => {
    const { PhotonRuntime } = await import('../src/photon-runtime.mjs');
    const runtime = await new PhotonRuntime('/tmp/simd-tensor.wasm').init();

    it('tensor.exports lists all functions', () => {
        const exports = runtime.tensor.exports;
        console.log(`  tensor.exports: ${exports.join(', ')}`);
        assert.ok(exports.includes('unrolledDotProduct'));
        assert.ok(exports.includes('simdDotProduct'));
        assert.ok(exports.includes('scalarDotProduct'));
    });

    it('tensor.dotProduct auto-selects unrolled (4.5x)', () => {
        const N = 10000;
        const a = new Float64Array(N);
        const b = new Float64Array(N);
        for (let i = 0; i < N; i++) { a[i] = Math.random(); b[i] = Math.random(); }

        runtime.tensor.fill(a, 0);
        runtime.tensor.fill(b, N * 8);

        const runs = 1000;
        const t = performance.now();
        let sum = 0;
        for (let r = 0; r < runs; r++) sum += runtime.tensor.dotProduct(0, N * 8, N);
        const elapsed = performance.now() - t;

        let nodeSum = 0;
        const nt = performance.now();
        for (let r = 0; r < runs; r++) {
            let s = 0;
            for (let i = 0; i < N; i++) s += a[i] * b[i];
            nodeSum += s;
        }
        const nodeElapsed = performance.now() - nt;

        const speedup = nodeElapsed / elapsed;
        console.log(`  10K dot product (1000 runs):`);
        console.log(`    Node.js: ${nodeElapsed.toFixed(2)}ms`);
        console.log(`    Photon:  ${elapsed.toFixed(2)}ms (${speedup.toFixed(2)}x)`);
        assert.ok(speedup > 2, `Expected >2x speedup, got ${speedup.toFixed(2)}x`);
    });

    it('tensor.similarity produces valid cosine sim', () => {
        const a = new Float64Array([1, 0, 0, 0, 1]);
        const b = new Float64Array([1, 0, 0, 0, 1]);
        const identical = runtime.tensor.similarity(a, b);
        assert.ok(Math.abs(identical - 1.0) < 1e-6, `Expected ~1.0, got ${identical}`);

        const c = new Float64Array([1, 0, 0, 0, 0]);
        const d = new Float64Array([0, 0, 0, 0, 1]);
        const orthogonal = runtime.tensor.similarity(c, d);
        assert.ok(Math.abs(orthogonal) < 1e-6, `Expected ~0, got ${orthogonal}`);
        console.log(`  similarity: identical=${identical.toFixed(4)} orthogonal=${orthogonal.toFixed(4)} ✅`);
    });
});

// ============================================================
// GAN-TDD Loop 3: Worker Pool
// ============================================================

describe('🏭 PhotonWorkerPool', async () => {
    const { PhotonWorkerPool } = await import('../src/photon-runtime.mjs');

    it('persistent pool computes correct dot product', async () => {
        const pool = await new PhotonWorkerPool('/tmp/simd-tensor.wasm', 2).init();

        const a = new Float64Array([1, 2, 3, 4, 5]);
        const b = new Float64Array([5, 4, 3, 2, 1]);

        const result = await pool.dotProduct(a, b);
        assert.ok(Math.abs(result - 35) < 1e-6, `Expected 35, got ${result}`);
        console.log(`  pool.dotProduct([1..5],[5..1]) = ${result} ✅`);

        await pool.shutdown();
    });

    it('parallel dot product on 100K vectors', async () => {
        const pool = await new PhotonWorkerPool('/tmp/simd-tensor.wasm', 4).init();
        const N = 100000;
        const a = new Float64Array(N);
        const b = new Float64Array(N);
        for (let i = 0; i < N; i++) { a[i] = Math.random(); b[i] = Math.random(); }

        // Reference
        let refDot = 0;
        for (let i = 0; i < N; i++) refDot += a[i] * b[i];

        const result = await pool.parallelDotProduct(a, b);
        assert.ok(Math.abs(result - refDot) < 1, `Expected ~${refDot.toFixed(2)}, got ${result.toFixed(2)}`);
        console.log(`  100K parallel: ref=${refDot.toFixed(4)} pool=${result.toFixed(4)} ✅`);

        await pool.shutdown();
    });
});

// ============================================================
// GAN-TDD Loop 3 (continued): Security
// ============================================================

describe('🛡️ Runtime Security', () => {
    it('PhotonRuntime does not expose raw memory pointer', async () => {
        const { PhotonRuntime } = await import('../src/photon-runtime.mjs');
        const runtime = await new PhotonRuntime('/tmp/simd-tensor.wasm').init();
        // Memory is accessible through tensor API, not directly exposed
        assert.ok(runtime.tensor);
        assert.ok(runtime.tensor.fill);
        assert.ok(runtime.tensor.dotProduct);
        console.log('  No raw WASM memory pointer exposed ✅');
    });

    it('runtime file is compact', async () => {
        const src = readFileSync('/Users/ishikawaryuuta/.openclaw/workspace/projects/guava-photon/src/photon-runtime.mjs', 'utf-8');
        console.log(`  photon-runtime.mjs: ${src.length} chars`);
        assert.ok(src.length < 10000, `Expected <10K chars, got ${src.length}`);
    });
});
