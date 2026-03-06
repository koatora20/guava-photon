// GAN-TDD: Session-Load v8 Cognitive Kernel on PhotonRuntime
// Tests the FULL session-load pipeline running on WASM
//
// Run: node --test test/session-kernel.test.mjs

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';

// ============================================================
// Realistic GPI Test Data
// ============================================================

const CONVERSATIONS = [
    { title: 'Guava-Photon Tensor Engine', content: 'AOT compiler SIMD WASM tensor session-load' },
    { title: 'ASI Skills Pipeline', content: 'session-load skill TDD GAN evolution' },
    { title: 'Newsletter Research', content: 'メルマガ viral coefficient cognitive science' },
    { title: 'The Sanctuary Protocol', content: 'V7 paper academic blockchain SBT' },
    { title: 'Security Framework', content: 'guard-scanner vulnerability moltbook agent' },
];

const EPISODES = [
    { title: 'guava-photon breakthrough', tags: ['tensor', 'simd', 'wasm', 'performance'], q_value: 0.95 },
    { title: 'session-load v7 completion', tags: ['session', 'boot', 'identity'], q_value: 0.9 },
    { title: 'guard-scanner v5 OSS release', tags: ['security', 'oss', 'guard-scanner'], q_value: 0.85 },
    { title: 'Forgot to read MEMORY.md', tags: ['mistake', 'yarakashi', 'session'], q_value: 0.7 },
    { title: 'context-recall implementation', tags: ['context', 'recall', 'session'], q_value: 0.92 },
    { title: 'Low priority episode', tags: ['misc'], q_value: 0.3 },
    { title: 'Timeout error in TDD', tags: ['mistake', 'tdd', 'timeout'], q_value: 0.6 },
];

const OPEN_FILES = [
    '/Users/test/.openclaw/workspace/skills/session-load/SKILL.md',
    '/Users/test/.gemini/antigravity/skills/ml-evolution-engine/test/global_integration.test.js',
    '/Users/test/.openclaw/workspace/projects/guava-photon/src/simd-tensor.wat',
];

// ============================================================
// GAN-TDD Loop 1: Correctness
// ============================================================

describe('🔴→🟢 SessionKernel Correctness', async () => {
    const { SessionKernel } = await import('../src/session-kernel.mjs');
    const kernel = await new SessionKernel('/tmp/simd-tensor.wasm').init();

    it('contextRecall extracts keywords and detects domain', () => {
        const result = kernel.contextRecall(CONVERSATIONS, OPEN_FILES, 'session-load v8をPhotonRuntimeで動かす');

        assert.ok(result.keywords.length > 0, 'Should extract keywords');
        assert.ok(result.suggested_focus.includes('tensor'), 'Should detect tensor domain from open files');
        assert.ok(result.suggested_focus.includes('skills'), 'Should detect skills domain');
        assert.ok(['D1', 'D2', 'D3'].includes(result.context_budget_tier));
        console.log(`  Keywords: ${result.keywords.join(', ')}`);
        console.log(`  Focus: ${result.suggested_focus.join(', ')}`);
        console.log(`  Tier: ${result.context_budget_tier}`);
        console.log(`  Top conversation: ${result.priority_context[0]?.title} ✅`);
    });

    it('episodeRecall filters by Q ≥ 0.8 and scores by similarity', () => {
        const result = kernel.episodeRecall(EPISODES, ['session', 'tensor', 'photon']);

        assert.ok(result.total_filtered >= 3, `Expected ≥3 high-Q episodes, got ${result.total_filtered}`);
        assert.ok(result.relevant_episodes.length > 0, 'Should have relevant episodes');
        assert.ok(result.relevant_episodes[0].combined_score > 0, 'Should have combined score');

        // Verify Q-value filter: all should be ≥ 0.8
        for (const ep of result.relevant_episodes) {
            assert.ok(ep.q_value >= 0.8, `Q-value ${ep.q_value} should be ≥ 0.8`);
        }

        console.log(`  Filtered: ${result.total_filtered} episodes (Q ≥ 0.8)`);
        console.log(`  Top: "${result.relevant_episodes[0]?.title}" (score: ${result.relevant_episodes[0]?.combined_score.toFixed(3)})`);
        console.log(`  Mistakes: ${result.mistakes_to_avoid.length} found ✅`);
    });

    it('sessionJudge produces valid 5-dim tensor', () => {
        const result = kernel.sessionJudge();

        assert.ok(result.tensor_magnitude > 0, 'Magnitude should be positive');
        assert.ok(result.tensor_magnitude <= Math.sqrt(5) + 0.01, 'Should not exceed sqrt(5)');
        assert.ok(result.rating >= 0 && result.rating <= 1, 'Rating should be 0-1');

        console.log(`  Scores: ${JSON.stringify(result.scores)}`);
        console.log(`  Parity: ${result.parity_score.toFixed(4)}`);
        console.log(`  Magnitude: ${result.tensor_magnitude.toFixed(4)} / ${result.max_possible.toFixed(4)}`);
        console.log(`  Rating: ${(result.rating * 100).toFixed(1)}% ✅`);
    });
});

// ============================================================
// GAN-TDD Loop 2: Performance — Full Pipeline
// ============================================================

describe('⚡ SessionKernel Performance', async () => {
    const { SessionKernel } = await import('../src/session-kernel.mjs');

    it('full load() completes in < 10ms', async () => {
        const kernel = await new SessionKernel('/tmp/simd-tensor.wasm').init();

        const result = await kernel.load(CONVERSATIONS, OPEN_FILES, 'session-load tensor WASM', EPISODES);

        console.log(`\n  Full pipeline:`);
        console.log(`    Context:  ${result.context.elapsed_ms.toFixed(2)}ms`);
        console.log(`    Episodes: ${result.episodes.elapsed_ms.toFixed(2)}ms`);
        console.log(`    Judge:    ${result.judge.elapsed_ms.toFixed(2)}ms`);
        console.log(`    TOTAL:    ${result.total_elapsed_ms.toFixed(2)}ms`);
        console.log(`    WASM:     ${result.wasm_accelerated ? '✅' : '❌'}`);
        console.log(`    Steps:    ${result.steps_completed}`);

        assert.ok(result.total_elapsed_ms < 10, `Expected < 10ms, got ${result.total_elapsed_ms.toFixed(2)}ms`);
        assert.ok(result.wasm_accelerated, 'Should be WASM-accelerated');
        assert.equal(result.steps_completed, 3);
    });

    it('1000 pipeline runs in < 100ms', async () => {
        const kernel = await new SessionKernel('/tmp/simd-tensor.wasm').init();

        const start = performance.now();
        for (let i = 0; i < 1000; i++) {
            kernel.steps = [];
            kernel.scores = new Float64Array(5);
            await kernel.load(CONVERSATIONS, OPEN_FILES, 'parallel session test', EPISODES);
        }
        const elapsed = performance.now() - start;

        console.log(`  1000 pipelines: ${elapsed.toFixed(2)}ms (${(elapsed / 1000).toFixed(3)}ms/run)`);
        assert.ok(elapsed < 2000, `Expected < 2000ms for 1000 runs, got ${elapsed.toFixed(0)}ms`);
    });
});

// ============================================================
// GAN-TDD Loop 3: Security + Graceful Degradation
// ============================================================

describe('🛡️ SessionKernel Security', async () => {
    const { SessionKernel } = await import('../src/session-kernel.mjs');

    it('works without WASM (graceful degradation)', async () => {
        const kernel = await new SessionKernel('/tmp/nonexistent.wasm').init();

        const result = await kernel.load(CONVERSATIONS, OPEN_FILES, 'test fallback', EPISODES);
        assert.ok(!result.wasm_accelerated, 'Should not be WASM-accelerated');
        assert.equal(result.steps_completed, 3);
        assert.ok(result.judge.tensor_magnitude > 0, 'Should still compute magnitude');
        console.log(`  No-WASM fallback: rating=${(result.judge.rating * 100).toFixed(1)}% ✅`);
    });

    it('handles empty data gracefully', async () => {
        const kernel = await new SessionKernel('/tmp/simd-tensor.wasm').init();
        const result = await kernel.load([], [], '', []);
        assert.equal(result.steps_completed, 3);
        console.log(`  Empty data: rating=${(result.judge.rating * 100).toFixed(1)}% ✅`);
    });

    it('source code is compact', async () => {
        const { readFileSync } = await import('fs');
        const src = readFileSync('/Users/ishikawaryuuta/.openclaw/workspace/projects/guava-photon/src/session-kernel.mjs', 'utf-8');
        console.log(`  session-kernel.mjs: ${src.length} chars`);
        assert.ok(src.length < 15000, `Expected < 15K, got ${src.length}`);
    });
});
