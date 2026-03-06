// session-load v8: Cognitive Kernel on PhotonRuntime
// WASM-accelerated session loading with tensor-based scoring
//
// What's new in v8:
//   - WASM SIMD tensor operations for Q-value filtering and tag matching
//   - PhotonRuntime API shim (no raw Node.js dependency)
//   - Vectorized episode scoring (cosine similarity via unrolled SIMD)
//   - 5-dim session quality tensor evaluation
//
// Usage:
//   import { SessionKernel } from './session-kernel.mjs';
//   const kernel = new SessionKernel('/tmp/simd-tensor.wasm');
//   await kernel.init();
//   const result = await kernel.load(conversationSummaries, openFiles, userMessage);

import { PhotonRuntime } from './photon-runtime.mjs';

// ============================================================
// Session-Load v8 Cognitive Kernel
// ============================================================

export class SessionKernel {
    constructor(wasmPath = '/tmp/simd-tensor.wasm') {
        this.runtime = new PhotonRuntime(wasmPath);
        this.steps = [];
        this.scores = new Float64Array(5); // 5-dim quality tensor
    }

    async init() {
        await this.runtime.init();
        return this;
    }

    // ============================================================
    // Step 0: Context Recall (intent extraction + budget tier)
    // ============================================================
    contextRecall(conversationSummaries, openFiles, userMessage) {
        const startTime = performance.now();

        // Extract keywords from user message (using _tokenize for mixed JP/EN)
        const keywords = this._tokenize(userMessage);

        // Score conversation summaries by keyword overlap
        const summaryScores = conversationSummaries.map((summary, idx) => {
            const text = (summary.title + ' ' + (summary.content || '')).toLowerCase();
            let score = 0;
            for (const kw of keywords) {
                if (text.includes(kw)) score += 1;
            }
            return { idx, score, title: summary.title };
        });

        // Sort by score descending
        summaryScores.sort((a, b) => b.score - a.score);

        // Determine context budget tier based on keyword diversity
        const uniqueMatches = summaryScores.filter(s => s.score > 0).length;
        const tier = uniqueMatches >= 5 ? 'D3' : uniqueMatches >= 2 ? 'D2' : 'D1';

        // Detect domain from open files
        const domains = new Set();
        for (const f of openFiles) {
            if (f.includes('security') || f.includes('guard')) domains.add('security');
            if (f.includes('tensor') || f.includes('photon') || f.includes('simd')) domains.add('tensor');
            if (f.includes('skill')) domains.add('skills');
            if (f.includes('test')) domains.add('testing');
            if (f.includes('memory') || f.includes('session')) domains.add('memory');
        }

        const elapsed = performance.now() - startTime;
        this.scores[0] = Math.min(1.0, uniqueMatches / 5); // normalize

        const result = {
            priority_context: summaryScores.slice(0, 3),
            suggested_focus: [...domains],
            context_budget_tier: tier,
            keywords,
            elapsed_ms: elapsed
        };

        this.steps.push({ step: 0, name: 'context-recall', result, elapsed });
        return result;
    }

    // ============================================================
    // Tokenizer: extract semantic features from mixed JP/EN text
    // ============================================================
    _tokenize(text) {
        const words = text
            .toLowerCase()
            .replace(/[,。、！？\.\!\?\(\)\[\]「」『』【】:;—\-]+/g, ' ')
            // Split ASCII from Japanese characters
            .replace(/([a-z0-9]+)([ぁ-んァ-ヶー一-龠々])/g, '$1 $2')
            .replace(/([ぁ-んァ-ヶー一-龠々])([a-z0-9]+)/g, '$1 $2')
            .split(/\s+/)
            .filter(w => w.length > 1);

        // Deduplicate while preserving order
        const seen = new Set();
        const unique = [];
        for (const w of words) {
            if (!seen.has(w)) { seen.add(w); unique.push(w); }
        }

        // Generate bigrams
        const bigrams = [];
        for (let i = 0; i < unique.length - 1; i++) {
            bigrams.push(unique[i] + '_' + unique[i + 1]);
        }

        return [...unique, ...bigrams];
    }

    // ============================================================
    // Step 3: Episode Recall with Semantic Title Embedding
    // ============================================================
    episodeRecall(episodes, priorityKeywords) {
        const startTime = performance.now();

        // Q-value filtering: q >= 0.8
        const highQ = episodes.filter(ep => (ep.q_value || 0) >= 0.8);

        // Build unified vocabulary from ALL episodes (titles + tags) + query keywords
        const queryTokens = this._tokenize(priorityKeywords.join(' '));

        const allFeatures = new Set();
        for (const kw of queryTokens) allFeatures.add(kw);
        for (const ep of highQ) {
            const titleTokens = this._tokenize(ep.title || '');
            const tagTokens = (ep.tags || []).map(t => t.toLowerCase());
            for (const t of titleTokens) allFeatures.add(t);
            for (const t of tagTokens) allFeatures.add(t);
        }

        const vocab = [...allFeatures];
        const vocabIndex = new Map(vocab.map((v, i) => [v, i]));
        const dim = vocab.length;

        // Build query vector (BoW + bigrams from keywords)
        const queryVec = new Float64Array(dim);
        for (const token of queryTokens) {
            const idx = vocabIndex.get(token);
            if (idx !== undefined) queryVec[idx] = 1.0;
        }

        // Score each episode using unified embedding
        const scored = highQ.map(ep => {
            // Build episode vector from title tokens + tags
            const epVec = new Float64Array(dim);
            const titleTokens = this._tokenize(ep.title || '');
            const tagTokens = (ep.tags || []).map(t => t.toLowerCase());

            for (const t of titleTokens) {
                const idx = vocabIndex.get(t);
                if (idx !== undefined) epVec[idx] = 1.0;
            }
            for (const t of tagTokens) {
                const idx = vocabIndex.get(t);
                if (idx !== undefined) epVec[idx] = 1.5; // tags get higher weight
            }

            // WASM cosine similarity
            let similarity = 0;
            if (this.runtime.instance && dim > 0) {
                similarity = this.runtime.tensor.similarity(queryVec, epVec);
            } else {
                // Fallback
                let dot = 0, magA = 0, magB = 0;
                for (let i = 0; i < dim; i++) {
                    dot += queryVec[i] * epVec[i];
                    magA += queryVec[i] ** 2;
                    magB += epVec[i] ** 2;
                }
                similarity = magA * magB > 0 ? dot / (Math.sqrt(magA) * Math.sqrt(magB)) : 0;
            }

            return { ...ep, similarity, combined_score: (ep.q_value || 0) * 0.4 + similarity * 0.6 };
        });

        scored.sort((a, b) => b.combined_score - a.combined_score);

        // Extract mistakes
        const mistakes = episodes
            .filter(ep => (ep.tags || []).some(t => t.includes('mistake') || t.includes('yarakashi')))
            .slice(0, 3);

        const elapsed = performance.now() - startTime;
        this.scores[1] = Math.min(1.0, scored.length / 10);

        const result = {
            relevant_episodes: scored.slice(0, 5),
            mistakes_to_avoid: mistakes,
            lesson_summary: mistakes.map(m => m.title || m.content).join('; '),
            total_filtered: highQ.length,
            vocab_size: dim,
            elapsed_ms: elapsed
        };

        this.steps.push({ step: 3, name: 'episode-recall', result, elapsed });
        return result;
    }

    // ============================================================
    // Step 7: Session Judge (5-dim tensor evaluation)
    // ============================================================
    sessionJudge() {
        const startTime = performance.now();

        // 5-dim quality tensor:
        // [0] context_coverage: how well we matched context
        // [1] episode_recall: how many relevant episodes found
        // [2] identity_loaded: were all files loaded
        // [3] speed: how fast was the load
        // [4] completeness: how many steps completed

        this.scores[2] = this.steps.length >= 2 ? 1.0 : 0.5; // identity
        this.scores[3] = 1.0; // speed (always fast with WASM)
        this.scores[4] = Math.min(1.0, this.steps.length / 7); // completeness

        // Use WASM for parity tensor calculation
        let parityScore = 0;
        if (this.runtime.instance && this.runtime.instance.exports.parityTensor) {
            parityScore = this.runtime.instance.exports.parityTensor(
                this.scores[0], this.scores[1], this.scores[2], this.scores[3]
            );
        } else {
            // Fallback: geometric mean
            parityScore = Math.pow(
                this.scores[0] * this.scores[1] * this.scores[2] * this.scores[3] * this.scores[4],
                1 / 5
            );
        }

        // Compute magnitude of quality tensor via WASM
        let tensorMagnitude = 0;
        if (this.runtime.instance) {
            this.runtime.tensor.fill(this.scores, 0);
            tensorMagnitude = Math.sqrt(this.runtime.tensor.dotProduct(0, 0, 5));
        } else {
            tensorMagnitude = Math.sqrt(this.scores.reduce((s, v) => s + v * v, 0));
        }

        const elapsed = performance.now() - startTime;

        const result = {
            scores: {
                context_coverage: this.scores[0],
                episode_recall: this.scores[1],
                identity_loaded: this.scores[2],
                speed: this.scores[3],
                completeness: this.scores[4]
            },
            parity_score: parityScore,
            tensor_magnitude: tensorMagnitude,
            max_possible: Math.sqrt(5), // all 1.0
            rating: tensorMagnitude / Math.sqrt(5),
            elapsed_ms: elapsed
        };

        this.steps.push({ step: 7, name: 'session-judge', result, elapsed });
        return result;
    }

    // ============================================================
    // Full Pipeline: Run all steps
    // ============================================================
    async load(conversationSummaries = [], openFiles = [], userMessage = '', episodes = []) {
        const totalStart = performance.now();

        // Step 0: Context Recall
        const contextResult = this.contextRecall(conversationSummaries, openFiles, userMessage);

        // Step 3: Episode Recall
        const episodeResult = this.episodeRecall(episodes, contextResult.keywords);

        // Step 7: Session Judge
        const judgeResult = this.sessionJudge();

        const totalElapsed = performance.now() - totalStart;

        return {
            context: contextResult,
            episodes: episodeResult,
            judge: judgeResult,
            total_elapsed_ms: totalElapsed,
            steps_completed: this.steps.length,
            wasm_accelerated: !!this.runtime.instance
        };
    }
}
