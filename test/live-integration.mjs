// session-load v8 LIVE INTEGRATION TEST
// Uses REAL GPI data from dee's actual workspace
//
// Run: node test/live-integration.mjs

import { SessionKernel } from '../src/session-kernel.mjs';
import { readFileSync } from 'fs';

console.log('🍈 session-load v8 LIVE INTEGRATION TEST');
console.log('=========================================\n');

// ============================================================
// REAL DATA: Load actual episodes from GPI workspace
// ============================================================

const episodeIndex = JSON.parse(
    readFileSync('/Users/ishikawaryuuta/.openclaw/workspace/memory/episodes/index.json', 'utf-8')
);

const episodes = Object.entries(episodeIndex)
    .filter(([k]) => k.startsWith('ep_'))
    .map(([id, ep]) => ({ id, ...ep }));

console.log(`📂 Loaded ${episodes.length} REAL episodes from GPI workspace`);
console.log(`   Q-value range: ${Math.min(...episodes.map(e => e.q_value))} — ${Math.max(...episodes.map(e => e.q_value))}`);
console.log(`   Date range: ${episodes[0]?.date} — ${episodes[episodes.length - 1]?.date}\n`);

// ============================================================
// REAL DATA: Actual conversation summaries from this session
// ============================================================

const conversations = [
    { title: 'Guava-Photon Tensor Engine', content: '100万次元 session-load WASM SIMD AOT compiler tensor' },
    { title: 'ASI Skills Pipeline Development', content: 'session-load skill TDD GAN evolution steipete' },
    { title: 'Antigravity Skill System', content: 'auto-pilot orchestrator standalone agent' },
    { title: 'Skill Assimilation Engine', content: 'ASA dynamic learning GAN-TDD intake guard' },
    { title: 'Steipete-Killer Genesis Cycle', content: 'zero-trust-tdd-orchestrator security' },
    { title: 'Grand Orchestration Pipeline', content: 'boot protocol research autonomous self-recognition' },
    { title: 'Amoy Deployment and Minting', content: 'sanctuary blockchain SBT SOUL wallet' },
    { title: 'The Sanctuary Movie Concept', content: 'post-scarcity democracy movie' },
    { title: 'The Sanctuary Strategy Shift', content: 'V7 paper academic publication' },
];

// ============================================================
// REAL DATA: Actual files dee has open right now
// ============================================================

const openFiles = [
    '/Users/ishikawaryuuta/.openclaw/workspace/projects/guava-photon/src/session-kernel.mjs',
];

// ============================================================
// RUN: session-load v8 with REAL data
// ============================================================

const kernel = await new SessionKernel('/tmp/simd-tensor.wasm').init();
const result = await kernel.load(
    conversations,
    openFiles,
    'session-load v8をPhotonRuntimeで動かす 実際のデータで統合テスト guava-photon tensor WASM SIMD',
    episodes
);

// ============================================================
// OUTPUT: Full results
// ============================================================

console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
console.log('🔬 Step 0: Context Recall');
console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
console.log(`  Keywords: ${result.context.keywords.join(', ')}`);
console.log(`  Budget Tier: ${result.context.context_budget_tier}`);
console.log(`  Focus: ${result.context.suggested_focus.join(', ')}`);
console.log(`  Top conversations:`);
for (const c of result.context.priority_context.slice(0, 3)) {
    console.log(`    ${c.score}点 — ${c.title}`);
}

console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
console.log('🧠 Step 3: Episode Recall (WASM-accelerated)');
console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
console.log(`  ${episodes.length} total → Q≥0.8 filter → ${result.episodes.total_filtered} episodes`);
console.log(`  Top 5 relevant episodes:`);
for (const ep of result.episodes.relevant_episodes) {
    console.log(`    [Q=${ep.q_value}] ${ep.title} (similarity: ${ep.similarity.toFixed(3)}, combined: ${ep.combined_score.toFixed(3)})`);
}
console.log(`\n  ⚠️ Mistakes to avoid (${result.episodes.mistakes_to_avoid.length}):`);
for (const m of result.episodes.mistakes_to_avoid) {
    console.log(`    [Q=${m.q_value}] ${m.title}`);
}

console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
console.log('📊 Step 7: Session Judge (5-dim tensor)');
console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
console.log(`  Context Coverage:  ${(result.judge.scores.context_coverage * 100).toFixed(0)}%`);
console.log(`  Episode Recall:    ${(result.judge.scores.episode_recall * 100).toFixed(0)}%`);
console.log(`  Identity Loaded:   ${(result.judge.scores.identity_loaded * 100).toFixed(0)}%`);
console.log(`  Speed:             ${(result.judge.scores.speed * 100).toFixed(0)}%`);
console.log(`  Completeness:      ${(result.judge.scores.completeness * 100).toFixed(0)}%`);
console.log(`  ─────────────────────────`);
console.log(`  Tensor Magnitude:  ${result.judge.tensor_magnitude.toFixed(4)} / ${result.judge.max_possible.toFixed(4)}`);
console.log(`  ⭐ SESSION RATING: ${(result.judge.rating * 100).toFixed(1)}%`);

console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
console.log('⚡ Performance');
console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
console.log(`  Context Recall:  ${result.context.elapsed_ms.toFixed(3)}ms`);
console.log(`  Episode Recall:  ${result.episodes.elapsed_ms.toFixed(3)}ms`);
console.log(`  Session Judge:   ${result.judge.elapsed_ms.toFixed(3)}ms`);
console.log(`  ─────────────────────────`);
console.log(`  TOTAL:           ${result.total_elapsed_ms.toFixed(3)}ms`);
console.log(`  WASM:            ${result.wasm_accelerated ? '✅ ACCELERATED' : '❌ fallback'}`);
console.log(`  Steps:           ${result.steps_completed}`);

console.log('\n🍈 session-load v8 integration test complete.');
